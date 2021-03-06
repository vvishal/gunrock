// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * partitioner_base.cuh
 *
 * @brief Base struct for all the partitioner types
 */

#pragma once

#include <gunrock/util/basic_utils.h>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/multithreading.cuh>

#include <vector>

namespace gunrock {
namespace app {

/**
 * @brief Base partitioner structure.
 *
 */

template <
    typename   _VertexId,
    typename   _SizeT,
    typename   _Value,
    bool       ENABLE_BACKWARD = false,
    bool       KEEP_ORDER      = false,
    bool       KEEP_NODE_NUM   = false>
struct PartitionerBase
{
    typedef _VertexId  VertexId;
    typedef _SizeT     SizeT;
    typedef _Value     Value;
    typedef Csr<VertexId,Value,SizeT> GraphT;

    // Members
public:
    // Number of GPUs to be partitioned
    int        num_gpus;
    int        Status;
    float      factor;
    int        seed;

    // Original graph
    const GraphT *graph;

    // Partioned graphs
    GraphT *sub_graphs;

    int           **partition_tables;
    VertexId      **convertion_tables;
    VertexId      **original_vertexes;
    int           **backward_partitions;
    VertexId      **backward_convertions;
    SizeT         **backward_offsets;
    SizeT         **in_counter;
    SizeT         **out_offsets;
    SizeT         **out_counter;

    //Mthods

    template <
        typename VertexId,
        typename SizeT,
        typename Value>
    struct ThreadSlice
    {
    public:
        const GraphT  *graph;
        GraphT        *sub_graph;
        GraphT        *sub_graphs;
        int           thread_num,num_gpus;
        util::cpu_mt::CPUBarrier* cpu_barrier;
        CUTThread     thread_Id;
        int           *partition_table0;
        int           **partition_table1;
        int           **partition_tables;
        VertexId      *convertion_table0;
        VertexId      **convertion_table1;
        VertexId      **convertion_tables;
        int           **backward_partitions;
        VertexId      **backward_convertions;
        SizeT         **backward_offsets;
        VertexId      **original_vertexes;
        SizeT         **in_counter;
        SizeT         **out_offsets;
        SizeT         **out_counter;
    };

    /**
     * @brief PartitionerBase default constructor
     */
    PartitionerBase()
    {
        Status               = 0;
        num_gpus             = 0;
        graph                = NULL;
        sub_graphs           = NULL;
        partition_tables     = NULL;
        convertion_tables    = NULL;
        original_vertexes    = NULL;
        in_counter           = NULL;
        out_offsets          = NULL;
        out_counter          = NULL;
        backward_partitions  = NULL;
        backward_convertions = NULL;
        backward_offsets     = NULL;
    }

    virtual ~PartitionerBase()
    {
        Release();        
    } 

    cudaError_t Init(
        const GraphT &graph,
        int   num_gpus)
    {   
        cudaError_t retval= cudaSuccess;
        this->num_gpus    = num_gpus;
        this->graph       = &graph;
        Release();

        sub_graphs        = new GraphT   [num_gpus  ];
        partition_tables  = new int*     [num_gpus+1];
        convertion_tables = new VertexId*[num_gpus+1];
        original_vertexes = new VertexId*[num_gpus  ];
        in_counter        = new SizeT*   [num_gpus  ];
        out_offsets       = new SizeT*   [num_gpus  ];
        out_counter       = new SizeT*   [num_gpus  ];
        if (ENABLE_BACKWARD)
        {
            backward_partitions  = new int*      [num_gpus];
            backward_convertions = new VertexId* [num_gpus];
            backward_offsets     = new SizeT*    [num_gpus];
        }
        
        for (int i=0;i<num_gpus+1;i++)
        {
            partition_tables [i] = NULL;
            convertion_tables[i] = NULL;
            if (i!=num_gpus) {
                original_vertexes   [i] = NULL;
                if (ENABLE_BACKWARD)
                {
                    backward_partitions [i] = NULL;
                    backward_convertions[i] = NULL;
                    backward_offsets    [i] = NULL;
                }
            }
        }
        partition_tables [0] = (int*) malloc(sizeof(int) * graph.nodes);
        convertion_tables[0] = (VertexId*) malloc(sizeof(VertexId) * graph.nodes);
        memset(partition_tables [0], 0, sizeof(int     ) * graph.nodes);
        memset(convertion_tables[0], 0, sizeof(VertexId) * graph.nodes);
        for (int i=0;i<num_gpus;i++)
        {
            in_counter   [i] = new SizeT [num_gpus+1];
            out_offsets  [i] = new SizeT [num_gpus+1];
            out_counter  [i] = new SizeT [num_gpus+1];
            memset(in_counter   [i], 0, sizeof(SizeT) * (num_gpus+1));
            memset(out_offsets  [i], 0, sizeof(SizeT) * (num_gpus+1)); 
            memset(out_counter  [i], 0, sizeof(SizeT) * (num_gpus+1));
        }
        Status = 1;

        return retval;
    }

    void Release()
    {
        if (Status==0) return;
        for (int i=0;i<num_gpus+1;i++)
        {
            free(convertion_tables [i]); convertion_tables [i] = NULL;
            free(partition_tables  [i]); partition_tables  [i] = NULL;
            if (i == num_gpus) continue;
            free(original_vertexes [i]); original_vertexes [i] = NULL;
            delete[] in_counter    [i] ; in_counter        [i] = NULL;
            delete[] out_offsets   [i] ; out_offsets       [i] = NULL;
            delete[] out_counter   [i] ; out_counter       [i] = NULL;
            if (ENABLE_BACKWARD)
            {
                free(backward_partitions [i]); backward_partitions [i] = NULL;
                free(backward_convertions[i]); backward_convertions[i] = NULL;
                free(backward_offsets    [i]); backward_offsets    [i] = NULL;
            }
        }
        delete[] convertion_tables; convertion_tables = NULL;
        delete[] partition_tables ; partition_tables  = NULL;
        delete[] original_vertexes; original_vertexes = NULL;
        if (num_gpus>1) delete[] sub_graphs       ; sub_graphs        = NULL;
        delete[] in_counter       ; in_counter        = NULL;
        delete[] out_offsets      ; out_offsets       = NULL;
        delete[] out_counter      ; out_counter       = NULL;
        if (ENABLE_BACKWARD)
        {
            delete[] backward_convertions; backward_convertions = NULL;
            delete[] backward_partitions ; backward_partitions  = NULL;
            delete[] backward_offsets    ; backward_offsets     = NULL;
        }
        Status = 0;
    }
    
    static CUT_THREADPROC MakeSubGraph_Thread(void *thread_data_)
    {
        ThreadSlice<VertexId,SizeT,Value> *thread_data = (ThreadSlice<VertexId,SizeT,Value> *) thread_data_;
        const GraphT*   graph                 = thread_data->graph;
        GraphT*         sub_graph             = thread_data->sub_graph;
        GraphT*         sub_graphs            = thread_data->sub_graphs;
        int             gpu                   = thread_data->thread_num;
        util::cpu_mt::CPUBarrier* cpu_barrier = thread_data->cpu_barrier;
        int             num_gpus              = thread_data->num_gpus;
        int*            partition_table0      = thread_data->partition_table0;
        VertexId*       convertion_table0     = thread_data->convertion_table0;
        int**           partition_table1      = thread_data->partition_table1;
        VertexId**      convertion_tables     = thread_data->convertion_tables;
        int**           partition_tables      = thread_data->partition_tables;
        VertexId**      convertion_table1     = thread_data->convertion_table1;
        VertexId**      original_vertexes     = thread_data->original_vertexes;
        int**           backward_partitions   = thread_data->backward_partitions;
        VertexId**      backward_convertions  = thread_data->backward_convertions;
        SizeT**         backward_offsets      = thread_data->backward_offsets;
        SizeT**         out_offsets           = thread_data->out_offsets;
        SizeT*          in_counter            = thread_data->in_counter [gpu];
        SizeT*          out_counter           = thread_data->out_counter[gpu];
        SizeT           num_nodes             = 0, node_counter;
        SizeT           num_edges             = 0, edge_counter;
        VertexId*       marker                = new VertexId[graph->nodes];
        VertexId*       tconvertion_table     = new VertexId[graph->nodes];
        SizeT           in_counter_           = 0;

        memset(marker, 0, sizeof(int)*graph->nodes);
        memset(out_counter, 0, sizeof(SizeT) * (num_gpus+1));

        for (SizeT node=0; node<graph->nodes; node++)
        if (partition_table0[node] == gpu)
        {
            convertion_table0[node] = KEEP_NODE_NUM ? node : out_counter[gpu];
            tconvertion_table[node] = KEEP_NODE_NUM ? node : out_counter[gpu];
            marker[node] =1;
            for (SizeT edge=graph->row_offsets[node]; edge<graph->row_offsets[node+1]; edge++)
            {
                SizeT neibor = graph->column_indices[edge];
                int peer  = partition_table0[neibor];
                if ((peer != gpu) && (marker[neibor] == 0))
                {
                    tconvertion_table[neibor] = KEEP_NODE_NUM ? neibor : out_counter[peer];
                    out_counter[peer]++;
                    marker[neibor]=1;
                    num_nodes++;
                }
            }
            out_counter[gpu]++;
            num_nodes++;
            num_edges+= graph->row_offsets[node+1] - graph->row_offsets[node];
        }
        delete[] marker;marker=NULL;
        out_offsets[gpu][0]=0;
        node_counter=out_counter[gpu];
        for (int peer=0;peer<num_gpus;peer++)
        {
            if (peer==gpu) continue;
            int peer_=peer < gpu? peer+1 : peer;
            out_offsets[gpu][peer_]=node_counter;
            node_counter+=out_counter[peer];
        }
        out_offsets[gpu][num_gpus]=node_counter;
        util::cpu_mt::PrintCPUArray<SizeT, SizeT>("out_offsets",out_offsets[gpu],num_gpus+1,gpu);
        util::cpu_mt::IncrementnWaitBarrier(cpu_barrier,gpu);
        
        node_counter=0;
        for (int peer=0;peer<num_gpus;peer++)
        {
            if (peer==gpu) continue;
            int peer_ = peer < gpu ? peer+1 : peer;
            int gpu_  = gpu  < peer? gpu +1 : gpu ; 
            in_counter[peer_]=out_offsets[peer][gpu_+1]-out_offsets[peer][gpu_];
            node_counter+=in_counter[peer_];
        }
        in_counter[num_gpus]=node_counter;

        if (KEEP_NODE_NUM) num_nodes = graph->nodes; 
        if      (graph->node_values == NULL && graph->edge_values == NULL) 
             sub_graph->template FromScratch < false , false  >(num_nodes,num_edges);
        else if (graph->node_values != NULL && graph->edge_values == NULL) 
             sub_graph->template FromScratch < false , true   >(num_nodes,num_edges);
        else if (graph->node_values == NULL && graph->edge_values != NULL) 
             sub_graph->template FromScratch < true  , false  >(num_nodes,num_edges);
        else sub_graph->template FromScratch < true  , true   >(num_nodes,num_edges);

        if (convertion_table1[0] != NULL) free(convertion_table1[0]);
        if (partition_table1 [0] != NULL) free(partition_table1 [0]);
        if (original_vertexes[0] != NULL) free(original_vertexes[0]);
        convertion_table1[0]= (VertexId*) malloc (sizeof(VertexId) * num_nodes);
        partition_table1 [0]= (int*)      malloc (sizeof(int)      * num_nodes);
        original_vertexes[0]= (VertexId*) malloc (sizeof(VertexId) * num_nodes);
        if (ENABLE_BACKWARD)
        {
            if (backward_partitions [gpu] !=NULL) free(backward_partitions [gpu]);
            if (backward_convertions[gpu] !=NULL) free(backward_convertions[gpu]);
            if (backward_offsets    [gpu] !=NULL) free(backward_offsets    [gpu]);
            backward_offsets    [gpu] = (SizeT*    ) malloc (sizeof(SizeT     ) * (num_nodes+1));
            backward_convertions[gpu] = (VertexId* ) malloc (sizeof(VertexId  ) * in_counter[num_gpus]);
            backward_partitions [gpu] = (int*      ) malloc (sizeof(int       ) * in_counter[num_gpus]);
            if (KEEP_NODE_NUM)
            {
                marker     = new VertexId[num_gpus * graph->nodes];
                memset(marker, 0, sizeof(VertexId) * num_gpus * graph->nodes);
            } else {
                marker     = new VertexId[num_gpus*out_counter[gpu]];
                memset(marker,0,sizeof(VertexId)*num_gpus*out_counter[gpu]);
            }
            for (SizeT neibor=0; neibor<graph->nodes; neibor++)
            if (partition_table0[neibor] != gpu)
            {
                for (SizeT edge = graph->row_offsets[neibor]; edge<graph->row_offsets[neibor+1]; edge++)
                {
                    VertexId node = graph->column_indices[edge];
                    if (partition_table0[node]!=gpu) continue;
                    marker[convertion_table0[node]*num_gpus + partition_table0[neibor]]=1+neibor;
                }
            }
        }
        edge_counter=0;
        for (SizeT node=0; node<graph->nodes; node++)
        if (partition_table0[node] == gpu)
        {
            VertexId node_ = tconvertion_table[node];
            sub_graph->row_offsets[node_]=edge_counter;
            if (graph->node_values != NULL) sub_graph->node_values[node_]=graph->node_values[node];
            partition_table1 [0][node_] = 0;
            convertion_table1[0][node_] = node_;
            original_vertexes[0][node_] = node;
            for (SizeT edge=graph->row_offsets[node]; edge<graph->row_offsets[node+1]; edge++)
            {
                SizeT    neibor  = graph->column_indices[edge];
                int      peer    = partition_table0[neibor];
                int      peer_   = peer < gpu ? peer+1 : peer;
                if (peer == gpu) peer_ = 0;
                VertexId neibor_ = KEEP_NODE_NUM ? neibor : tconvertion_table[neibor] + out_offsets[gpu][peer_];
                
                sub_graph->column_indices[edge_counter] = neibor_;
                if (graph->edge_values !=NULL) sub_graph->edge_values[edge_counter]=graph->edge_values[edge];
                if (peer != gpu && !KEEP_NODE_NUM)
                {
                    sub_graph->row_offsets[neibor_]=num_edges;
                    partition_table1 [0][neibor_] = peer_;
                    convertion_table1[0][neibor_] = convertion_table0[neibor];
                    original_vertexes[0][neibor_] = neibor;
                }
                edge_counter++;
            }   
        } else if (KEEP_NODE_NUM) {
            sub_graph->row_offsets[node] = edge_counter;
            partition_table1 [0][node] = partition_table0 [node] < gpu? partition_table0[node]+1: partition_table0[node];
            convertion_table1[0][node] = convertion_table0[node];
            original_vertexes[0][node] = node;
        }
        sub_graph->row_offsets[num_nodes]=num_edges;

        if (ENABLE_BACKWARD)
        {
            in_counter_ = 0;
            util::cpu_mt::IncrementnWaitBarrier(cpu_barrier,gpu);
            if (!KEEP_NODE_NUM) 
            {
                for (VertexId node_=0; node_<num_nodes; node_++)
                {
                    backward_offsets[gpu][node_]=in_counter_;
                    if (partition_table1[0][node_]!=0) 
                    {
                        continue;
                    }
                    for (int peer=0;peer<num_gpus;peer++)
                    {
                        if (marker[node_*num_gpus+peer]==0) continue;
                        int  peer_ = peer < gpu ? peer+1 : peer;
                        int  gpu_  = gpu  < peer ? gpu+1 : gpu;
                        VertexId neibor = marker[node_*num_gpus+peer]-1;
                        VertexId neibor_ = convertion_table0[neibor];
                        for (SizeT edge=sub_graphs[peer].row_offsets[neibor_]; edge<sub_graphs[peer].row_offsets[neibor_+1];edge++)
                        {
                            VertexId _node = sub_graphs[peer].column_indices[edge];
                            if (convertion_tables[peer+1][_node] == node_ && 
                                partition_tables [peer+1][_node] == gpu_)
                            {
                                backward_convertions[gpu][in_counter_]= _node;
                                break;
                            }
                        }
                        backward_partitions[gpu][in_counter_]=peer_;
                        in_counter_++;
                    }
                }
                backward_offsets[gpu][num_nodes]=in_counter_;
            } else {
                delete[] backward_partitions[gpu];backward_partitions[gpu] = new int[num_nodes * (num_gpus-1)];
                delete[] backward_convertions[gpu];backward_convertions[gpu] = new VertexId[num_nodes * (num_gpus-1)];
                for (VertexId node=0; node<num_nodes; node++)
                {
                    backward_offsets[gpu][node]=node*(num_gpus-1);
                    for (int peer=1;peer<num_gpus;peer++)
                    {
                        backward_convertions[gpu][node*(num_gpus-1)+peer-1]=node;
                        backward_partitions [gpu][node*(num_gpus-1)+peer-1]=peer;
                    }
                }
                backward_offsets[gpu][num_nodes]=num_nodes*(num_gpus-1);
            }
            delete[] marker;marker=NULL;
        } 
        out_counter[num_gpus]=0;
        in_counter[num_gpus]=0;
        for (int peer=0;peer<num_gpus;peer++)
        {
            int peer_ = peer < gpu? peer+1:peer;
            int gpu_  = peer < gpu? gpu: gpu+1;
            if (peer==gpu) {peer_=0;gpu_=0;}
            out_counter[peer_]=out_offsets[gpu][peer_+1]-out_offsets[gpu][peer_];
            out_counter[num_gpus]+=out_counter[peer_];
            in_counter[peer_]=out_offsets[peer][gpu_+1]-out_offsets[peer][gpu_];
            in_counter[num_gpus]+=in_counter[peer_];
        }
        util::cpu_mt::PrintCPUArray<SizeT, SizeT>("out_counter",out_counter,num_gpus+1,gpu);
        util::cpu_mt::PrintCPUArray<SizeT, SizeT>("in_counter ", in_counter,num_gpus+1,gpu);
        delete[] tconvertion_table; tconvertion_table = NULL;
        CUT_THREADEND;
    }

    cudaError_t MakeSubGraph()
    {
        cudaError_t retval = cudaSuccess;
        ThreadSlice<VertexId,SizeT,Value>* thread_data = new ThreadSlice<VertexId,SizeT,Value>[num_gpus];
        CUTThread*   thread_Ids  = new CUTThread  [num_gpus];
        util::cpu_mt::CPUBarrier   cpu_barrier;
        cpu_barrier = util::cpu_mt::CreateBarrier(num_gpus);

        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            thread_data[gpu].graph               = graph;
            thread_data[gpu].sub_graph           = &(sub_graphs[gpu]);
            thread_data[gpu].sub_graphs          = sub_graphs;
            thread_data[gpu].thread_num          = gpu;
            thread_data[gpu].cpu_barrier         = &cpu_barrier;
            thread_data[gpu].num_gpus            = num_gpus;
            thread_data[gpu].partition_table0    = partition_tables [0];
            thread_data[gpu].convertion_table0   = convertion_tables[0];
            thread_data[gpu].partition_tables    = partition_tables;
            thread_data[gpu].partition_table1    = &(partition_tables[gpu+1]);
            thread_data[gpu].convertion_table1   = &(convertion_tables[gpu+1]);
            thread_data[gpu].original_vertexes   = &(original_vertexes[gpu]);
            thread_data[gpu].convertion_tables   = convertion_tables;
            if (ENABLE_BACKWARD) {
                thread_data[gpu].backward_partitions  = backward_partitions ;
                thread_data[gpu].backward_convertions = backward_convertions;
                thread_data[gpu].backward_offsets     = backward_offsets    ;
            }
            thread_data[gpu].in_counter          = in_counter;
            thread_data[gpu].out_offsets         = out_offsets;
            thread_data[gpu].out_counter         = out_counter;
            thread_data[gpu].thread_Id           = cutStartThread((CUT_THREADROUTINE)&(MakeSubGraph_Thread), (void*)(&(thread_data[gpu])));
            thread_Ids[gpu]=thread_data[gpu].thread_Id;
        }

        cutWaitForThreads(thread_Ids,num_gpus);

        util::cpu_mt::DestoryBarrier(&cpu_barrier);
        delete[] thread_Ids ;thread_Ids =NULL;
        delete[] thread_data;thread_data=NULL;
        Status = 2;
        return retval;
    }

    cudaError_t Partition(
        GraphT*    &sub_graphs,
        int**      &partition_tables,
        VertexId** &convertion_tables,
        VertexId** &original_vertexes,
        SizeT**    &out_offsets,
        SizeT**    &cross_counter,
        float      factor = -1,
        int        seed = -1)
    {   
        SizeT**    backward_offsets     = NULL;
        int**      backward_partitions  = NULL;
        VertexId** backward_convertions = NULL;
        return Partition(
                   sub_graphs,
                   partition_tables,
                   convertion_tables,
                   original_vertexes,
                   in_counter,
                   out_offsets,
                   out_counter,
                   backward_offsets,
                   backward_partitions,
                   backward_convertions,
                   factor,
                   seed);
    }   

    virtual cudaError_t Partition(
        GraphT*    &sub_graphs,
        int**      &partition_tables,
        VertexId** &convertion_tables,
        VertexId** &original_vertexes,
        SizeT**    &in_counter,
        SizeT**    &out_offsets,
        SizeT**    &out_counter,
        SizeT**    &backward_offsets,
        int**      &backward_partitions,
        VertexId** &backward_convertions,
        float      factor = -1,
        int        seed = -1)
    {
        return util::GRError("PartitionBase::Partition is undefined", __FILE__, __LINE__);
    }
};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
