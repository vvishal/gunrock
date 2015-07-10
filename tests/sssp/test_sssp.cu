// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_sssp.cu
 *
 * @brief Simple test driver program for single source shorest path.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>
#include <gunrock/graphio/rmat.cuh>
#include <gunrock/graphio/rgg.cuh>

// SSSP includes
#include <gunrock/app/sssp/sssp_enactor.cuh>
#include <gunrock/app/sssp/sssp_problem.cuh>
#include <gunrock/app/sssp/sssp_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/priority_queue/kernel.cuh>

#include <moderngpu.cuh>

// Boost includes for CPU dijkstra SSSP reference algorithms
#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::sssp;


/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

//bool g_verbose;
//bool g_undirected;
//bool g_quick;
//bool g_stream_from_host;

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
void Usage()
{
    printf(
        " test_sssp <graph type> <graph type args> [--device=<device_index>]\n"
        " [--undirected] [--instrumented] [--src=<source index>] [--quick=<0|1>]\n"
        " [--mark-pred] [--queue-sizing=<scale factor>] [--traversal-mode=<0|1>]\n"
        " [--in-sizing=<in/out queue scale factor>] [--disable-size-check]\n"
        " [--grid-size=<grid size>] [partition_method=<random|biasrandom|clustered|metis>]\n"
        " [--v] [--iteration-num=<num>]\n"
        "\n"
        "Graph types and args:\n"
        "  market [<file>]\n"
        "    Reads a Matrix-Market coordinate-formatted graph of directed / undirected\n"
        "    edges from stdin (or from the optionally-specified file).\n"
        "  --device=<device_index>   Set GPU device for running the test. [Default: 0].\n"
        "  --undirected              Treat the graph as undirected (symmetric).\n"
        "  --instrumented            Keep kernels statics [Default: Disable].\n"
        "                            total_queued, search_depth and barrier duty\n"
        "                            (a relative indicator of load imbalance.)\n"
        "  --src=<source vertex id>  Begins SSSP from the source [Default: 0].\n"
        "                            If randomize: from a random source vertex.\n"
        "                            If largestdegree: from largest degree vertex.\n"
        "  --quick=<0 or 1>          Skip the CPU validation: 1, or not: 0 [Default: 1].\n"
        "  --mark-pred               Keep both label info and predecessor info.\n"
        "  --queue-sizing=<factor>   Allocates a frontier queue sized at:\n"
        "                            (graph-edges * <scale factor>) [Default: 1.0].\n"
        "  --v                       Print verbose per iteration debug info.\n"
        "  --iteration-num=<number>  Number of runs to perform the test [Default: 1].\n"
        "  --traversal-mode=<0 or 1> Set traversal strategy, 0 for Load-Balanced,\n"
        "                            1 for Dynamic-Cooperative [Default: dynamic\n"
        "                            determine based on average degree].\n"
        );
}

/**
 * @brief Displays the SSSP result (i.e., distance from source)
 *
 * @param[in] source_path Search depth from the source for each node.
 * @param[in] nodes Number of nodes in the graph.
 */
template<typename VertexId, typename SizeT>
void DisplaySolution (VertexId *source_path, SizeT num_nodes)
{
    if (num_nodes > 40) num_nodes = 40;

    printf("[");
    for (VertexId i = 0; i < num_nodes; ++i)
    {
        PrintValue(i);
        printf(":");
        PrintValue(source_path[i]);
        printf(" ");
    }
    printf("]\n");
}

/**
 * Performance/Evaluation statistics
 */

struct Stats {
    const char *name;
    Statistic rate;
    Statistic search_depth;
    Statistic redundant_work;
    Statistic duty;

    Stats() : name(NULL), rate(), search_depth(), redundant_work(), duty() {}
    Stats(const char *name) : name(name), rate(), search_depth(), redundant_work(), duty() {}
};

struct Test_Parameter : gunrock::app::TestParameter_Base {
public:
    //bool          mark_predecessors ;// Whether or not to mark src-distance vs. parent vertices
    int delta_factor;
    double max_queue_sizing1;

    Test_Parameter()
    { 
        delta_factor = 16;
        mark_predecessors = false;
        max_queue_sizing1 = -1.0;
    }   

    ~Test_Parameter()
    {   
    }   

    void Init(CommandLineArgs &args)
    {   
        TestParameter_Base::Init(args);
        mark_predecessors = args.CheckCmdLineFlag("mark-pred");
        args.GetCmdLineArgument("delta-factor"    , delta_factor    );
        args.GetCmdLineArgument("queue-sizing1", max_queue_sizing1);
    }   
};

/**
 * @brief Displays timing and correctness statistics
 *
 * @tparam MARK_PREDECESSORS
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] stats Reference to the Stats object defined in RunTests
 * @param[in] src Source node where SSSP starts
 * @param[in] h_labels Host-side vector stores computed labels for validation
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] elapsed Total elapsed kernel running time
 * @param[in] search_depth Maximum search depth of the SSSP algorithm
 * @param[in] total_queued Total element queued in SSSP kernel running process
 * @param[in] avg_duty Average duty of the SSSP kernels
 */
template<
    typename VertexId,
    typename Value,
    typename SizeT>
void DisplayStats(
    Stats               &stats,
    VertexId            src,
    Value               *h_labels,
    const Csr<VertexId, Value, SizeT> &graph,
    double              elapsed,
    VertexId            search_depth,
    long long           total_queued,
    double              avg_duty)
{
    // Compute nodes and edges visited
    SizeT edges_visited = 0;
    SizeT nodes_visited = 0;
    for (VertexId i = 0; i < graph.nodes; ++i) {
        if (h_labels[i] < util::MaxValue<VertexId>()) {
            ++nodes_visited;
            edges_visited += graph.row_offsets[i+1] - graph.row_offsets[i];
        }
    }

    double redundant_work = 0.0;
    if (total_queued > 0)
    {
        redundant_work =
            ((double) total_queued - edges_visited) / edges_visited;
    }
    redundant_work *= 100;

    // Display test name
    printf("[%s] finished.", stats.name);

    // Display statistics
    if (nodes_visited < 5)
    {
        printf("Fewer than 5 vertices visited.\n");
    }
    else
    {
        // Display the specific sample statistics
        double m_teps = (double) edges_visited / (elapsed * 1000.0);
        printf("\n elapsed: %.4f ms, rate: %.4f MiEdges/s", elapsed, m_teps);
        if (search_depth != 0)
            printf(", search_depth: %lld", (long long) search_depth);
        printf("\n src: %lld, nodes_visited: %lld, edges_visited: %lld",
               (long long) src, (long long) nodes_visited, (long long) edges_visited);
        if (avg_duty != 0)
        {
            printf("\n avg CTA duty: %.2f%%", avg_duty * 100);
        }
        if (total_queued > 0)
        {
            printf(", total queued: %lld", total_queued);
        }
        if (redundant_work > 0)
        {
            printf(", redundant work: %.2f%%", redundant_work);
        }
        printf("\n");
    }
}

/******************************************************************************
 * SSSP Testing Routines
 *****************************************************************************/

/**
 * @brief A simple CPU-based reference SSSP ranking implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] node_values Host-side vector to store CPU computed labels for each node
 * @param[in] node_preds Host-side vector to store CPU computed predecessors for each node
 * @param[in] src Source node where SSSP starts
 */
template<
    typename VertexId,
    typename Value,
    typename SizeT,
    bool     MARK_PREDECESSORS>
void SimpleReferenceSssp(
    const Csr<VertexId, Value, SizeT>       &graph,
    Value                                   *node_values,
    VertexId                                *node_preds,
    VertexId                                src)
{
    using namespace boost;

    // Prepare Boost Datatype and Data structure
    typedef adjacency_list<vecS, vecS, directedS, no_property,
                           property <edge_weight_t, unsigned int> > Graph;

    typedef graph_traits<Graph>::vertex_descriptor vertex_descriptor;
    typedef graph_traits<Graph>::edge_descriptor edge_descriptor;

    typedef std::pair<unsigned int, unsigned int> Edge;

    Edge* edges = (Edge*)malloc(sizeof(Edge)*graph.edges);
    unsigned int *weight =
        (unsigned int*)malloc(sizeof(unsigned int)*graph.edges);

    for (int i = 0; i < graph.nodes; ++i)
    {
        for (int j = graph.row_offsets[i]; j < graph.row_offsets[i+1]; ++j)
        {
            edges[j] = Edge(i, graph.column_indices[j]);
            weight[j] = graph.edge_values[j];
        }
    }

    Graph g(edges, edges + graph.edges, weight, graph.nodes);

    std::vector<unsigned int> d(graph.nodes);
    std::vector<vertex_descriptor> p(graph.nodes);
    vertex_descriptor s = vertex(src, g);

    property_map<Graph, vertex_index_t>::type indexmap = get(vertex_index, g);

    //
    // Perform SSSP
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();

    if (MARK_PREDECESSORS)
        dijkstra_shortest_paths(
            g, s,
            predecessor_map(boost::make_iterator_property_map(p.begin(), get(boost::vertex_index, g))).
            distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, g))));
    else
        dijkstra_shortest_paths(
            g, s,
            distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, g))));
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    printf("CPU SSSP finished in %lf msec.\n", elapsed);

    Coo<unsigned int, unsigned int>* sort_dist = NULL;
    Coo<unsigned int, unsigned int>* sort_pred = NULL;
    sort_dist = (Coo<unsigned int, unsigned int>*)malloc(
        sizeof(Coo<unsigned int, unsigned int>) * graph.nodes);
    if (MARK_PREDECESSORS)
        sort_pred = (Coo<unsigned int, unsigned int>*)malloc(
            sizeof(Coo<unsigned int, unsigned int>) * graph.nodes);

    graph_traits < Graph >::vertex_iterator vi, vend;
    for (tie(vi, vend) = vertices(g); vi != vend; ++vi)
    {
        sort_dist[(*vi)].row = (*vi);
        sort_dist[(*vi)].col = d[(*vi)];
    }
    std::stable_sort(
        sort_dist, sort_dist + graph.nodes,
        RowFirstTupleCompare<Coo<unsigned int, unsigned int> >);

    if (MARK_PREDECESSORS)
    {
        for (tie(vi, vend) = vertices(g); vi != vend; ++vi)
        {
            sort_pred[(*vi)].row = (*vi);
            sort_pred[(*vi)].col = p[(*vi)];
        }
        std::stable_sort(
            sort_pred, sort_pred + graph.nodes,
            RowFirstTupleCompare<Coo<unsigned int, unsigned int> >);
    }

    for (int i = 0; i < graph.nodes; ++i)
    {
        node_values[i] = sort_dist[i].col;
    }
    if (MARK_PREDECESSORS)
        for (int i = 0; i < graph.nodes; ++i)
        {
            node_preds[i] = sort_pred[i].col;
        }

    free(sort_dist);
    if (MARK_PREDECESSORS) free(sort_pred);
}


/**
 * @brief Run SSSP tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] src Source node where SSSP starts
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] queue_sizing Scaling factor used in edge mapping
 * @param[in] num_gpus Number of GPUs
 * @param[in] delta_factor Parameter to specify delta in delta-stepping SSSP
 * @param[in] iterations Number of iteration for running the test
 & @param[in] traversal_mode Load-balanced or Dynamic cooperative
 * @param[in] context CudaContext pointer for moderngpu APIs
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT,
    bool DEBUG,
    bool SIZE_CHECK,
    bool MARK_PREDECESSORS>
void RunTests(Test_Parameter *parameter)
{
    typedef SSSPProblem<
        VertexId,
        SizeT,
        Value,
        MARK_PREDECESSORS> Problem;

    typedef SSSPEnactor<
        Problem,
        INSTRUMENT,
        DEBUG,
        SIZE_CHECK> Enactor;

    Csr<VertexId, Value, SizeT>
                 *graph                 = (Csr<VertexId, Value, SizeT>*)parameter->graph;
    VertexId      src                   = (VertexId)parameter -> src;
    int           max_grid_size         = parameter -> max_grid_size;
    int           num_gpus              = parameter -> num_gpus;
    double        max_queue_sizing      = parameter -> max_queue_sizing;
    double        max_in_sizing         = parameter -> max_in_sizing;
    ContextPtr   *context               = (ContextPtr*)parameter -> context;
    std::string   partition_method      = parameter -> partition_method;
    int          *gpu_idx               = parameter -> gpu_idx;
    cudaStream_t *streams               = parameter -> streams;
    float         partition_factor      = parameter -> partition_factor;
    int           partition_seed        = parameter -> partition_seed;
    bool          g_quick               = parameter -> g_quick;
    bool          g_stream_from_host    = parameter -> g_stream_from_host;
    int           delta_factor          = parameter -> delta_factor;
    int           iterations            = parameter -> iterations;
    int           traversal_mode        = parameter -> traversal_mode;
    size_t       *org_size              = new size_t[num_gpus];
    // Allocate host-side label array (for both reference and gpu-computed results)
    Value        *reference_labels      = new Value[graph->nodes];
    Value        *h_labels              = new Value[graph->nodes];
    Value        *reference_check_label = (g_quick) ? NULL : reference_labels;
    VertexId     *reference_preds       = MARK_PREDECESSORS ? new VertexId[graph->nodes] : NULL;
    VertexId     *h_preds               = MARK_PREDECESSORS ? new VertexId[graph->nodes] : NULL;
    VertexId     *reference_check_pred  = (g_quick || !MARK_PREDECESSORS) ? NULL : reference_preds;

    for (int gpu=0;gpu<num_gpus;gpu++)
    {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]),&dummy);
    }
        
    // Allocate SSSP enactor map
    Enactor* enactor = new Enactor(num_gpus, gpu_idx);

    // Allocate problem on GPU
    Problem *problem = new Problem;
    util::GRError(problem->Init(
        g_stream_from_host,
        graph,
        NULL,
        num_gpus,
        gpu_idx,
        partition_method,
        streams,
        delta_factor,
        max_queue_sizing,
        max_in_sizing,
        partition_factor,
        partition_seed), "Problem SSSP Initialization Failed", __FILE__, __LINE__);
    util::GRError(enactor->Init (context, problem, max_grid_size, traversal_mode), "SSSP Enactor init failed", __FILE__, __LINE__);
    //
    // Compute reference CPU SSSP solution for source-distance
    //
    if (reference_check_label != NULL)
    {
        printf("Computing reference value ...\n");
        SimpleReferenceSssp<VertexId, Value, SizeT, MARK_PREDECESSORS>(
                *graph,
                reference_check_label,
                reference_check_pred,
                src);
        printf("\n");
    }

    Stats      *stats       = new Stats("GPU SSSP");
    long long  total_queued = 0;
    VertexId   search_depth = 0;
    double     avg_duty     = 0.0;
    float      elapsed      = 0.0f;

    // Perform SSSP
    CpuTimer cpu_timer;

    for (int iter = 0; iter < iterations; ++iter)
    {
        util::GRError(problem->Reset(src, enactor->GetFrontierType(), max_queue_sizing), "SSSP Problem Data Reset Failed", __FILE__, __LINE__); 
        util::GRError(enactor->Reset(), "SSSP Enactor Reset failed", __FILE__, __LINE__);

        printf("__________________________\n");fflush(stdout);
        cpu_timer.Start();
        util::GRError(enactor->Enact(src, traversal_mode), "SSSP Problem Enact Failed", __FILE__, __LINE__);
        cpu_timer.Stop();
        printf("--------------------------\n");fflush(stdout);
        elapsed += cpu_timer.ElapsedMillis();
    }
    elapsed /= iterations;

    enactor->GetStatistics(total_queued, search_depth, avg_duty);

    // Copy out results
    util::GRError(problem->Extract(h_labels, h_preds), "SSSP Problem Data Extraction Failed", __FILE__, __LINE__);

    for (SizeT i=0; i<graph->nodes;i++)
    if (reference_check_label[i]==-1) reference_check_label[i]=util::MaxValue<Value>();

    // Display Solution
    printf("\nFirst 40 labels of the GPU result.\n"); 
    DisplaySolution(h_labels, graph->nodes);
 
    // Verify the result
    if (reference_check_label != NULL) {
        printf("Label Validity: ");
        int error_num = CompareResults(h_labels, reference_check_label, graph->nodes, true);
        if (error_num > 0)
            printf("%d errors occurred.\n", error_num);
        printf("\nFirst 40 labels of the reference CPU result.\n"); 
        DisplaySolution(reference_check_label, graph->nodes);
    }
    
    if (MARK_PREDECESSORS) {
        printf("\nFirst 40 preds of the GPU result.\n"); 
        DisplaySolution(h_preds, graph->nodes);
        if (reference_check_label != NULL) 
        {
            printf("\nFirst 40 preds of the reference CPU result (could be different because the paths are not unique).\n"); 
            DisplaySolution(reference_check_pred, graph->nodes);
        }
    }

    DisplayStats(
        *stats,
        src,
        h_labels,
        *graph,
        elapsed,
        search_depth,
        total_queued,
        avg_duty);

    printf("\n\tMemory Usage(B)\t");
    for (int gpu=0;gpu<num_gpus;gpu++)
    if (num_gpus>1) {if (gpu!=0) printf(" #keys%d,0\t #keys%d,1\t #ins%d,0\t #ins%d,1",gpu,gpu,gpu,gpu); else printf(" #keys%d,0\t #keys%d,1", gpu, gpu);}
    else printf(" #keys%d,0\t #keys%d,1", gpu, gpu);
    if (num_gpus>1) printf(" #keys%d",num_gpus);
    printf("\n");
    double max_queue_sizing_[2] = {0,0}, max_in_sizing_=0;
    for (int gpu=0;gpu<num_gpus;gpu++)
    {   
        size_t gpu_free,dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&gpu_free,&dummy);
        printf("GPU_%d\t %ld",gpu_idx[gpu],org_size[gpu]-gpu_free);
        for (int i=0;i<num_gpus;i++)
        {   
            for (int j=0; j<2; j++)
            {   
                SizeT x=problem->data_slices[gpu]->frontier_queues[i].keys[j].GetSize();
                printf("\t %lld", (long long) x); 
                double factor = 1.0*x/(num_gpus>1?problem->graph_slices[gpu]->in_counter[i]:problem->graph_slices[gpu]->nodes);
                if (factor > max_queue_sizing_[j]) max_queue_sizing_[j]=factor;
            }   
            if (num_gpus>1 && i!=0 )
            for (int t=0;t<2;t++)
            {   
                SizeT x=problem->data_slices[gpu][0].keys_in[t][i].GetSize();
                printf("\t %lld", (long long) x); 
                double factor = 1.0*x/problem->graph_slices[gpu]->in_counter[i];
                if (factor > max_in_sizing_) max_in_sizing_=factor;
            }   
        }   
        if (num_gpus>1) printf("\t %lld", (long long)(problem->data_slices[gpu]->frontier_queues[num_gpus].keys[0].GetSize()));
        printf("\n");
    }   
    printf("\t queue_sizing =\t %lf \t %lf", max_queue_sizing_[0], max_queue_sizing_[1]);
    if (num_gpus>1) printf("\t in_sizing =\t %lf", max_in_sizing_);
    printf("\n");

    // Cleanup
    if (org_size        ) {delete[] org_size        ; org_size         = NULL;}
    if (stats           ) {delete   stats           ; stats            = NULL;}
    if (enactor         ) {delete   enactor         ; enactor          = NULL;}
    if (problem         ) {delete   problem         ; problem          = NULL;}
    if (reference_labels) {delete[] reference_labels; reference_labels = NULL;}
    if (h_labels        ) {delete[] h_labels        ; h_labels         = NULL;}
    if (reference_preds ) {delete[] reference_preds ; reference_preds  = NULL;}
    if (h_preds         ) {delete[] h_preds         ; h_preds          = NULL;}

    //cudaDeviceSynchronize();
}

template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT,
    bool        DEBUG,
    bool        SIZE_CHECK>
void RunTests_mark_predecessors(Test_Parameter *parameter)
{
    if (parameter->mark_predecessors) RunTests
        <VertexId, Value, SizeT, INSTRUMENT, DEBUG, SIZE_CHECK,
        true > (parameter);
   else RunTests
        <VertexId, Value, SizeT, INSTRUMENT, DEBUG, SIZE_CHECK,
        false> (parameter);
}

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT,
    bool          INSTRUMENT,
    bool          DEBUG>
void RunTests_size_check(Test_Parameter *parameter)
{
    if (parameter->size_check) RunTests_mark_predecessors
        <VertexId, Value, SizeT, INSTRUMENT, DEBUG,
        true > (parameter);
   else RunTests_mark_predecessors
        <VertexId, Value, SizeT, INSTRUMENT, DEBUG,
        false> (parameter);
}

template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT>
void RunTests_debug(Test_Parameter *parameter)
{
    if (parameter->debug) RunTests_size_check
        <VertexId, Value, SizeT, INSTRUMENT,
        true > (parameter);
    else RunTests_size_check
        <VertexId, Value, SizeT, INSTRUMENT,
        false> (parameter);
}

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT>
void RunTests_instrumented(Test_Parameter *parameter)
{
    if (parameter->instrumented) RunTests_debug
        <VertexId, Value, SizeT,
        true > (parameter);
    else RunTests_debug
        <VertexId, Value, SizeT,
        false> (parameter);
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] args Reference to the command line arguments
 * @param[in] context CudaContext pointer for moderngpu APIs
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT>
void RunTests(
    Csr<VertexId, Value, SizeT> *graph,
    CommandLineArgs             &args,
    int                         num_gpus,
    ContextPtr                  *context,
    int                         *gpu_idx,
    cudaStream_t                *streams)
{
    string src_str = "";
    Test_Parameter *parameter = new Test_Parameter;
    
    parameter -> Init(args);
    parameter -> graph              = graph;
    parameter -> num_gpus           = num_gpus;
    parameter -> context            = context;
    parameter -> gpu_idx            = gpu_idx;
    parameter -> streams            = streams;

    args.GetCmdLineArgument("src", src_str);
    if (src_str.empty()) {
        parameter->src = 0;
    } else if (src_str.compare("randomize") == 0) {
        parameter->src = graphio::RandomNode(graph->nodes);
    } else if (src_str.compare("largestdegree") == 0) {
        int max_degree;
        parameter->src = graph->GetNodeWithHighestDegree(max_degree);
        printf("Using highest degree (%d) vertex: %d\n", max_degree, parameter->src);
    } else {
        args.GetCmdLineArgument("src", parameter->src);
    }

    // traversal mode
    args.GetCmdLineArgument("traversal-mode", parameter->traversal_mode);
    if (parameter->traversal_mode == -1)
    {
        parameter->traversal_mode = graph->GetAverageDegree() > 8 ? 0 : 1;
    }

    printf("src = %lld\n", parameter->src);

    RunTests_instrumented<VertexId, Value, SizeT>(parameter);
}

/******************************************************************************
* Main
******************************************************************************/

int main( int argc, char** argv)
{
    CommandLineArgs args(argc, argv);
    int          num_gpus = 0;
    int          *gpu_idx = NULL;
    ContextPtr   *context = NULL;
    cudaStream_t *streams = NULL;
    bool          g_undirected = false;

    if ((argc < 2) || (args.CheckCmdLineFlag("help"))) {
        Usage();
        return 1;
    }

    if (args.CheckCmdLineFlag  ("device"))
    {
        std::vector<int> gpus;
        args.GetCmdLineArguments<int>("device",gpus);
        num_gpus   = gpus.size();
        gpu_idx    = new int[num_gpus];
        for (int i=0;i<num_gpus;i++)
            gpu_idx[i] = gpus[i];
    } else {
        num_gpus   = 1;
        gpu_idx    = new int[num_gpus];
        gpu_idx[0] = 0;
    }
    streams  = new cudaStream_t[num_gpus * num_gpus *2];
    context  = new ContextPtr  [num_gpus * num_gpus];
    printf("Using %d gpus: ", num_gpus);
    for (int gpu=0;gpu<num_gpus;gpu++)
    {
        printf(" %d ", gpu_idx[gpu]);
        util::SetDevice(gpu_idx[gpu]);
        for (int i=0;i<num_gpus*2;i++)
        {
            int _i=gpu*num_gpus*2+i;
            util::GRError(cudaStreamCreate(&streams[_i]), "cudaStreamCreate fialed.",__FILE__,__LINE__);
            if (i<num_gpus) context[gpu*num_gpus+i] = mgpu::CreateCudaDeviceAttachStream(gpu_idx[gpu],streams[_i]);
        }
    }
    printf("\n"); fflush(stdout);
    
    // Parse graph-contruction params
    g_undirected = args.CheckCmdLineFlag("undirected");
    std::string graph_type = argv[1];
    int flags = args.ParsedArgc();
    int graph_args = argc - flags - 1;

    if (graph_args < 1) {
        Usage();
        return 1;
    }
	
    //
    // Construct graph and perform search(es)
    //
    typedef int VertexId;                   // Use as the node identifier type
    typedef int Value;                      // Use as the value type
    typedef long long  int SizeT;                      // Use as the graph size type
    Csr<VertexId, Value, SizeT> csr(false); // default value for stream_from_host is false
    if (graph_args < 1) { Usage(); return 1; }

    if (graph_type == "market") {
    // Matrix-market coordinate-formatted graph file

        char *market_filename = (graph_args == 2) ? argv[2] : NULL;
        if (graphio::BuildMarketGraph<true>(
            market_filename, 
            csr, 
            g_undirected,
            false) != 0) // no inverse graph
        {
            return 1;
        }

    } else if (graph_type == "rmat")
    {   
        // parse rmat parameters
        SizeT rmat_nodes = 1 << 10; 
        SizeT rmat_edges = 1 << 10; 
        SizeT rmat_scale = 10; 
        SizeT rmat_edgefactor = 48; 
        double rmat_a = 0.57;
        double rmat_b = 0.19;
        double rmat_c = 0.19;
        double rmat_d = 1-(rmat_a+rmat_b+rmat_c);
        double rmat_vmultipiler = 20;
        double rmat_vmin        = 1;
        int    rmat_seed        = -1;

        args.GetCmdLineArgument("rmat_scale", rmat_scale);
        rmat_nodes = 1 << rmat_scale;
        args.GetCmdLineArgument("rmat_nodes", rmat_nodes);
        args.GetCmdLineArgument("rmat_edgefactor", rmat_edgefactor);
        rmat_edges = rmat_nodes * rmat_edgefactor;
        rmat_vmultipiler = rmat_edgefactor * 2;
        args.GetCmdLineArgument("rmat_edges", rmat_edges);
        args.GetCmdLineArgument("rmat_a", rmat_a);
        args.GetCmdLineArgument("rmat_b", rmat_b);
        args.GetCmdLineArgument("rmat_c", rmat_c);
        rmat_d = 1-(rmat_a+rmat_b+rmat_c);
        args.GetCmdLineArgument("rmat_d", rmat_d);
        args.GetCmdLineArgument("rmat_vmultipiler", rmat_vmultipiler);
        args.GetCmdLineArgument("rmat_vmin", rmat_vmin);
        args.GetCmdLineArgument("rmat_seed", rmat_seed);

        CpuTimer cpu_timer;
        cpu_timer.Start();
        if (graphio::BuildRmatGraph<true>(
                rmat_nodes,
                rmat_edges,
                csr,
                g_undirected,
                rmat_a,
                rmat_b,
                rmat_c,
                rmat_d,
                rmat_vmultipiler,
                rmat_vmin,
                rmat_seed) != 0)
        {   
            return 1;
        }   
        cpu_timer.Stop();
        float elapsed = cpu_timer.ElapsedMillis();
        printf("graph generated: %.3f ms, a = %.3f, b = %.3f, c = %.3f, d = %.3f\n", elapsed, rmat_a, rmat_b, rmat_c, rmat_d);
    } else if (graph_type == "rgg") {
    
        SizeT rgg_nodes = 1 << 10; 
        SizeT rgg_scale = 10; 
        double rgg_thfactor  = 0.55;
        double rgg_threshold = rgg_thfactor * sqrt(log(rgg_nodes) / rgg_nodes);
        double rgg_vmultipiler = 20;
        double rgg_vmin = 1;
        int    rgg_seed = -1;
    
        args.GetCmdLineArgument("rgg_scale", rgg_scale);
        rgg_nodes = 1 << rgg_scale;
        args.GetCmdLineArgument("rgg_nodes", rgg_nodes);
        args.GetCmdLineArgument("rgg_thfactor", rgg_thfactor);
        rgg_threshold = rgg_thfactor * sqrt(log(rgg_nodes) / rgg_nodes);
        args.GetCmdLineArgument("rgg_threshold", rgg_threshold);
        args.GetCmdLineArgument("rgg_vmultipiler", rgg_vmultipiler);
        args.GetCmdLineArgument("rgg_vmin", rgg_vmin);
        args.GetCmdLineArgument("rgg_seed", rgg_seed);

        CpuTimer cpu_timer;
        cpu_timer.Start();
        if (graphio::BuildRggGraph<true>(
            rgg_nodes,
            csr,
            rgg_threshold,
            g_undirected,
            rgg_vmultipiler,
            rgg_vmin,
            rgg_seed) !=0)
        {
            return 1;
        }
        cpu_timer.Stop();
        float elapsed = cpu_timer.ElapsedMillis();
        printf("graph generated: %.3f ms, threshold = %.3lf, vmultipiler = %.3lf\n", elapsed, rgg_threshold, rgg_vmultipiler);
    } else {
        // Unknown graph type
        fprintf(stderr, "Unspecified graph type\n");
        return 1;
    }

    csr.PrintHistogram();
    csr.DisplayGraph(true); //print graph with edge_value
    //util::cpu_mt::PrintCPUArray("row_offsets", csr.row_offsets,csr.nodes+1);
    //util::cpu_mt::PrintCPUArray("colum_indiece", csr.column_indices, csr.edges);
    
    csr.GetAverageEdgeValue();
    csr.GetAverageDegree();
    int max_degree;
    csr.GetNodeWithHighestDegree(max_degree);
    printf("max degree:%d\n", max_degree);

    // Run tests
    RunTests(&csr, args, num_gpus, context, gpu_idx, streams);

    return 0;
}
