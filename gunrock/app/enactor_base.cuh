// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_base.cuh
 *
 * @brief Base Graph Problem Enactor
 */

#pragma once
#include <time.h>

#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/app/problem_base.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <moderngpu.cuh>

#ifdef WITHMPI
#include <mpi.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#endif

using namespace mgpu;

namespace gunrock {
namespace app {


	template <typename SizeT1, typename SizeT2>
	__global__ void Accumulate_Num (
	                                SizeT1 *num,
	                                SizeT2 *sum)
	{
	    sum[0]+=num[0];
	}

	/*
	 * @brief Structure for auxiliary variables used in enactor.
	 */
	struct EnactorStats
	{
	    long long                        iteration           ;
	    unsigned long long               total_lifetimes     ;
	    unsigned long long               total_runtimes      ;
	    util::Array1D<int, size_t>       total_queued        ;
	    unsigned int                     advance_grid_size   ;
	    unsigned int                     filter_grid_size    ;
	    util::KernelRuntimeStatsLifetime advance_kernel_stats;
	    util::KernelRuntimeStatsLifetime filter_kernel_stats ;
	    util::Array1D<int, unsigned int> node_locks          ;
	    util::Array1D<int, unsigned int> node_locks_out      ;
	    cudaError_t                      retval              ;
	    clock_t                          start_time          ;
    
	    EnactorStats()
	    {
	        iteration       = 0;
	        total_lifetimes = 0;
	        total_runtimes  = 0;
	        retval          = cudaSuccess;
	        node_locks    .SetName("node_locks"    );
	        node_locks_out.SetName("node_locks_out");
	        total_queued  .SetName("total_queued");
	    }
    
	    template <typename SizeT2>
	    void Accumulate(SizeT2 *d_queued, cudaStream_t stream)
	    {
	        Accumulate_Num<<<1,1,0,stream>>> (d_queued, total_queued.GetPointer(util::DEVICE));
	    }
	};

	/*
	 * @brief Structure for auxiliary variables used in frontier operations.
	 */
	template <typename SizeT>
	struct FrontierAttribute
	{
	    SizeT        queue_length ;
	    util::Array1D<SizeT,SizeT>
	    output_length;
	    unsigned int queue_index  ;
	    SizeT        queue_offset ;
	    int          selector     ;
	    bool         queue_reset  ;
	    int          current_label;
	    bool         has_incoming ;
	    gunrock::oprtr::advance::TYPE
	    advance_type ;
    
	    FrontierAttribute()
	    {
	        queue_length  = 0;
	        queue_index   = 0;
	        queue_offset  = 0;
	        selector      = 0;
	        queue_reset   = false;
	        has_incoming  = false;
	        output_length.SetName("output_length");
	    }
	};

	class ThreadSlice
	{
	public:
	    int           thread_num ;
	    int           init_size  ;
	    CUTThread     thread_Id  ;
	    int           stats      ;
	    void         *problem    ;
	    void         *enactor    ;
	    ContextPtr   *context    ;
	    util::cpu_mt::CPUBarrier
	    *cpu_barrier;
    
	    ThreadSlice()
	    {
	        problem     = NULL;
	        enactor     = NULL;
	        context     = NULL;
	        thread_num  = 0;
	        init_size   = 0;
	        stats       = -2;
	        cpu_barrier = NULL;
	    }
    
	    virtual ~ThreadSlice()
	    {
	        problem     = NULL;
	        enactor     = NULL;
	        context     = NULL;
	        cpu_barrier = NULL;
	    }
	};


	template <typename SizeT, typename DataSlice>
	bool All_Done(EnactorStats                    *enactor_stats,
	              FrontierAttribute<SizeT>        *frontier_attribute,
	              util::Array1D<SizeT, DataSlice> *data_slice,
	              int                              num_gpus,
	              int                              num_gpus_global
	              )
	{
	    //check for sane enactor_stats
	    for (int gpu=0;gpu<num_gpus*num_gpus_global;gpu++)
	        if (enactor_stats[gpu].retval!=cudaSuccess)
	        {
	            printf("(CUDA error %d @ GPU %d: %s\n", enactor_stats[gpu].retval, gpu%num_gpus, cudaGetErrorString(enactor_stats[gpu].retval)); fflush(stdout);
	            return true;
	        }
    
	    //check if any local data slice is to be continued
	    for (int gpu=0;gpu<num_gpus*num_gpus;gpu++)
	        if (frontier_attribute[gpu].queue_length!=0 || frontier_attribute[gpu].has_incoming)
	        {
	            //printf("frontier_attribute[%d].queue_length = %d\n",gpu,frontier_attribute[gpu].queue_length);
	            return false;
	        }
    
	    //check if any input buffers are still non-empty
	    for (int gpu=0;gpu<num_gpus;gpu++)
	        for (int peer=1;peer<num_gpus_global;peer++)
	            for (int i=0;i<2;i++)
	                if (data_slice[gpu]->in_length[i][peer]!=0)
	                {
	                    //printf("data_slice[%d]->in_length[%d][%d] = %d\n", gpu, i, peer, data_slice[gpu]->in_length[i][peer]);
	                    return false;
	                }
    
	    //check if any ouput buffers are still non-empty
	    for (int gpu=0;gpu<num_gpus;gpu++)
	        for (int peer=1;peer<num_gpus_global;peer++)
	            if (data_slice[gpu]->out_length[peer]!=0)
	            {
	                //printf("data_slice[%d]->out_length[%d] = %d\n", gpu, peer, data_slice[gpu]->out_length[peer]);
	                return false;
	            }
    
	    return true;
	}


	template <typename VertexId, typename SizeT>
	__global__ void Copy_Preds (
	                            const SizeT     num_elements,
	                            const VertexId* keys,
	                            const VertexId* in_preds,
	                            VertexId* out_preds)
	{
	    const SizeT STRIDE = gridDim.x * blockDim.x;
	    VertexId x = blockIdx.x*blockDim.x+threadIdx.x;
	    VertexId t;
    
	    while (x<num_elements)
	    {
	        t = keys[x];
	        out_preds[t] = in_preds[t];
	        x+= STRIDE;
	    }
	}

	template <typename VertexId, typename SizeT>
	__global__ void Update_Preds (
	                              const SizeT     num_elements,
	                              const SizeT     nodes,
	                              const VertexId* keys,
	                              const VertexId* org_vertexs,
	                              const VertexId* in_preds,
	                              VertexId* out_preds)
	{
	    const SizeT STRIDE = gridDim.x * blockDim.x;
	    VertexId x = blockIdx.x*blockDim.x + threadIdx.x;
	    VertexId t, p;
    
	    while (x<num_elements)
	    {
	        t = keys[x];
	        p = in_preds[t];
	        if (p<nodes) out_preds[t] = org_vertexs[p];
	        x+= STRIDE;
	    }
	}

	template <typename VertexId, typename SizeT>
	__global__ void Assign_Marker(
	                              const SizeT            num_elements,
	                              const int              num_gpus,
	                              const VertexId* const  keys_in,
	                              const int*      const  partition_table,
	                              SizeT**          marker)
	{
	    VertexId key;
	    int gpu;
	    extern __shared__ SizeT* s_marker[];
	    const SizeT STRIDE = gridDim.x * blockDim.x;
	    SizeT x= blockIdx.x * blockDim.x + threadIdx.x;
	    if (threadIdx.x < num_gpus)
	        s_marker[threadIdx.x]=marker[threadIdx.x];
	    __syncthreads();
    
	    while (x < num_elements)
	    {
	        key = keys_in[x];
	        gpu = partition_table[key];
	        for (int i=0;i<num_gpus;i++)
	            s_marker[i][x]=(i==gpu)?1:0;
	        x+=STRIDE;
	    }
	}

	template <typename VertexId, typename SizeT>
	__global__ void Assign_Marker_Backward(
	                                       const SizeT            num_elements,
	                                       const int              num_gpus,
	                                       const VertexId* const  keys_in,
	                                       const SizeT*    const  offsets,
	                                       const int*      const  partition_table,
	                                       SizeT**          marker)
	{
	    VertexId key;
	    extern __shared__ SizeT* s_marker[];
	    const SizeT STRIDE = gridDim.x * blockDim.x;
	    SizeT x= blockIdx.x * blockDim.x + threadIdx.x;
	    if (threadIdx.x < num_gpus)
	        s_marker[threadIdx.x]=marker[threadIdx.x];
	    __syncthreads();
    
	    while (x < num_elements)
	    {
	        key = keys_in[x];
	        for (int gpu=0;gpu<num_gpus;gpu++)
	            s_marker[gpu][x]=0;
	        if (key!=-1) for (SizeT i=offsets[key];i<offsets[key+1];i++)
	            s_marker[partition_table[i]][x]=1;
	        x+=STRIDE;
	    }
	}

	template <typename VertexId, typename SizeT, typename Value,
	SizeT num_vertex_associates, SizeT num_value__associates>
	__global__ void Make_Out(
	                         const  SizeT             num_elements,
	                         const  int               num_gpus,
	                         const  VertexId*   const keys_in,
	                         const  int*        const partition_table,
	                         const  VertexId*   const convertion_table,
	                         const  size_t            array_size,
	                         char*             array)
	{
	    extern __shared__ char s_array[];
	    const SizeT STRIDE = gridDim.x * blockDim.x;
	    size_t     offset                  = 0;
	    SizeT**    s_marker                = (SizeT**   )&(s_array[offset]);
	    offset+=sizeof(SizeT*   )*num_gpus;
	    VertexId** s_keys_outs             = (VertexId**)&(s_array[offset]);
	    offset+=sizeof(VertexId*)*num_gpus;
	    VertexId** s_vertex_associate_orgs = (VertexId**)&(s_array[offset]);
	    offset+=sizeof(VertexId*)*num_vertex_associates;
	    Value**    s_value__associate_orgs = (Value**   )&(s_array[offset]);
	    offset+=sizeof(Value*   )*num_value__associates;
	    VertexId** s_vertex_associate_outss= (VertexId**)&(s_array[offset]);
	    offset+=sizeof(VertexId*)*num_gpus*num_vertex_associates;
	    Value**    s_value__associate_outss= (Value**   )&(s_array[offset]);
	    offset+=sizeof(Value*   )*num_gpus*num_value__associates;
	    SizeT*     s_offset                = (SizeT*    )&(s_array[offset]);
	    SizeT x= threadIdx.x;
    
	    while (x<array_size)
	    {
	        s_array[x]=array[x];
	        x+=blockDim.x;
	    }
	    __syncthreads();
    
	    x= blockIdx.x * blockDim.x + threadIdx.x;
	    while (x<num_elements)
	    {
	        VertexId key    = keys_in [x];
	        int      target = partition_table[key];
	        SizeT    pos    = s_marker[target][x]-1 + s_offset[target];
        
	        if (target==0)
	        {
	            s_keys_outs[0][pos]=key;
	        } else {
	            s_keys_outs[target][pos]=convertion_table[key];
	#pragma unrool
	            for (int i=0;i<num_vertex_associates;i++)
	                s_vertex_associate_outss[target*num_vertex_associates+i][pos]
	                =s_vertex_associate_orgs[i][key];
	#pragma unrool
	            for (int i=0;i<num_value__associates;i++)
	                s_value__associate_outss[target*num_value__associates+i][pos]
	                =s_value__associate_orgs[i][key];
	        }
	        x+=STRIDE;
	    }
	}

	template <typename VertexId, typename SizeT, typename Value,
	SizeT num_vertex_associates, SizeT num_value__associates>
	__global__ void Make_Out_Backward(
	                                  const  SizeT             num_elements,
	                                  const  int               num_gpus,
	                                  const  VertexId*   const keys_in,
	                                  const  SizeT*      const offsets,
	                                  const  int*        const partition_table,
	                                  const  VertexId*   const convertion_table,
	                                  const  size_t            array_size,
	                                  char*             array)
	{
	    extern __shared__ char s_array[];
	    const SizeT STRIDE = gridDim.x * blockDim.x;
	    size_t     offset                  = 0;
	    SizeT**    s_marker                = (SizeT**   )&(s_array[offset]);
	    offset+=sizeof(SizeT*   )*num_gpus;
	    VertexId** s_keys_outs             = (VertexId**)&(s_array[offset]);
	    offset+=sizeof(VertexId*)*num_gpus;
	    VertexId** s_vertex_associate_orgs = (VertexId**)&(s_array[offset]);
	    offset+=sizeof(VertexId*)*num_vertex_associates;
	    Value**    s_value__associate_orgs = (Value**   )&(s_array[offset]);
	    offset+=sizeof(Value*   )*num_value__associates;
	    VertexId** s_vertex_associate_outss= (VertexId**)&(s_array[offset]);
	    offset+=sizeof(VertexId*)*num_gpus*num_vertex_associates;
	    Value**    s_value__associate_outss= (Value**   )&(s_array[offset]);
	    offset+=sizeof(Value*   )*num_gpus*num_value__associates;
	    SizeT*     s_offset                = (SizeT*    )&(s_array[offset]);
	    SizeT x= threadIdx.x;
    
	    while (x<array_size)
	    {
	        s_array[x]=array[x];
	        x+=blockDim.x;
	    }
	    __syncthreads();
    
	    x= blockIdx.x * blockDim.x + threadIdx.x;
	    while (x<num_elements)
	    {
	        VertexId key    = keys_in [x];
	        if (key <0) {x+=STRIDE; continue;}
	        for (SizeT j=offsets[key];j<offsets[key+1];j++)
	        {
	            int      target = partition_table[j];
	            SizeT    pos    = s_marker[target][x]-1 + s_offset[target];
            
	            if (target==0)
	            {
	                s_keys_outs[0][pos]=key;
	            } else {
	                s_keys_outs[target][pos]=convertion_table[j];
	#pragma unrool
	                for (int i=0;i<num_vertex_associates;i++)
	                    s_vertex_associate_outss[target*num_vertex_associates+i][pos]
	                    =s_vertex_associate_orgs[i][key];
	#pragma unrool
	                for (int i=0;i<num_value__associates;i++)
	                    s_value__associate_outss[target*num_value__associates+i][pos]
	                    =s_value__associate_orgs[i][key];
	            }
	        }
	        x+=STRIDE;
	    }
	}

	template <typename VertexId, typename SizeT>
	__global__ void Mark_Queue (
	                            const SizeT     num_elements,
	                            const VertexId* keys,
	                            unsigned int* marker)
	{
	    VertexId x = ((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
	    if (x< num_elements) marker[keys[x]]=1;
	}

	template <
	bool     SIZE_CHECK,
	typename SizeT,
	typename Type>
	cudaError_t Check_Size(
	                       const char *name,
	                       SizeT       target_length,
	                       util::Array1D<SizeT, Type>
	                       *array,
	                       bool       &oversized,
	                       int         thread_num = -1,
	                       int         iteration  = -1,
	                       int         peer_      = -1,
	                       bool        keep_content = false)
	{
	    cudaError_t retval = cudaSuccess;
    
	    if (target_length > array->GetSize())
	    {
	        printf("%d\t %d\t %d\t %s \t oversize :\t %d ->\t %d\n",
	               thread_num, iteration, peer_, name,
	               array->GetSize(), target_length);
	        fflush(stdout);
	        oversized=true;
	        if (SIZE_CHECK)
	        {
	            if (array->GetSize() != 0) retval = array->EnsureSize(target_length, keep_content);
	            else retval = array->Allocate(target_length, util::DEVICE);
	        } else {
	            char temp_str[]=" oversize", str[256];
	            memcpy(str, name, sizeof(char) * strlen(name));
	            memcpy(str + strlen(name), temp_str, sizeof(char) * strlen(temp_str));
	            str[strlen(name)+strlen(temp_str)]='0';
	            retval = util::GRError(cudaErrorLaunchOutOfResources, str, __FILE__, __LINE__);
	        }
	    }
	    return retval;
	}


	#ifdef WITHMPI
	template <
	bool     SIZE_CHECK,
	typename SizeT,
	typename VertexId,
	typename Value,
	typename GraphSlice,
	typename DataSlice,
	SizeT    num_vertex_associate,
	SizeT    num_value__associate>
	void MPI_PushNeighbor(
	                      int gpu,
	                      int peer,
	                      int iteration,
	                      int peer_server_idx,
	                      SizeT             queue_length,
	                      EnactorStats      *enactor_stats,
	                      DataSlice         *data_slice_l,
	                      DataSlice         *data_slice_p,
	                      GraphSlice        *graph_slice_l,
	                      GraphSlice        *graph_slice_p,
	                      MPI_Request ** sent_requests,
	                      MPI_Datatype mpi_vertex_type,
	                      MPI_Datatype mpi_value__type)
	{
	    if (peer == gpu) return;
	    int gpu_  = peer<gpu? gpu : gpu+1;
	    int peer_ = peer<gpu? peer+1 : peer;
	    int header[4];
	    header[0] = gpu;
	    header[1] = iteration;
	    header[2] = queue_length;
	    header[3] = gpu_;
	    if(MPI_SUCCESS != MPI_Isend(header, 3, MPI_INT, peer_server_idx, 0, MPI_COMM_WORLD,&(sent_requests[peer][0])))
	    {
	        printf("Unable to sent header to server %i\n", peer_server_idx);
	        return;
	    }
    
	    int sent_flag_counter=0;
	    if(MPI_SUCCESS != MPI_Isend(
	                                data_slice_l -> keys_out[peer_].GetPointer(util::DEVICE),
	                                queue_length,
	                                mpi_vertex_type,
	                                peer_server_idx,
	                                gpu,
	                                MPI_COMM_WORLD,
	                                &(sent_requests[gpu][sent_flag_counter])))
	    {
	        fprintf(stderr,"MPI_Isend error in MPI_PushNeighbor %s:%d\n",__FILE__,__LINE__);
	        return;
	    }
	    sent_flag_counter++;
    
	    for (int i=0;i<num_vertex_associate;i++)
	    {
	        if(MPI_SUCCESS != MPI_Isend(
	                                    data_slice_l->vertex_associate_outs[peer_][i],
	                                    queue_length,
	                                    mpi_vertex_type,
	                                    peer_server_idx,
	                                    gpu,
	                                    MPI_COMM_WORLD,
	                                    &(sent_requests[gpu][sent_flag_counter])))
	        {
	            fprintf(stderr,"MPI_Isend error in MPI_PushNeighbor %s:%d\n",__FILE__,__LINE__);
	            return;
	        }
	        sent_flag_counter++;
	    }
    
	    for (int j=0;j<num_value__associate;j++)
	    {
	        if(MPI_SUCCESS != MPI_Isend(
	                                    data_slice_l->value__associate_outs[peer_][j],
	                                    queue_length,
	                                    mpi_vertex_type,
	                                    peer_server_idx,
	                                    gpu,
	                                    MPI_COMM_WORLD,
	                                    &(sent_requests[gpu][sent_flag_counter])))
	        {
	            fprintf(stderr,"MPI_Isend error in MPI_PushNeighbor %s:%d\n",__FILE__,__LINE__);
	            return;
	        }
	        sent_flag_counter++;
	    }
    
	}


	int MPI_Communication_Check(MPI_Request ** sent_requests, int peer, int num_gpus_global, int num_flags)
	{
	    MPI_Status status;
	    int flag;
	    if(peer > num_gpus_global) return 0;
	    for(int f=0; f<num_flags; f++){
	        MPI_Test(&(sent_requests[peer][f]), &flag, &status);
	        if(flag==0) return 0;
	    }
	    return 1;
	}


	#endif

	template <
	bool     SIZE_CHECK,
	typename SizeT,
	typename VertexId,
	typename Value,
	typename GraphSlice,
	typename DataSlice,
	SizeT    num_vertex_associate,
	SizeT    num_value__associate>
	void PushNeighbor(
	                  int gpu,
	                  int peer,
	                  SizeT             queue_length,
	                  EnactorStats      *enactor_stats,
	                  DataSlice         *data_slice_l,
	                  DataSlice         *data_slice_p,
	                  GraphSlice        *graph_slice_l,
	                  GraphSlice        *graph_slice_p,
	                  cudaStream_t      stream)
	{
	    if (peer == gpu) return;
	    int gpu_  = peer<gpu? gpu : gpu+1;
	    int peer_ = peer<gpu? peer+1 : peer;
	    int i, t  = enactor_stats->iteration%2;
	    bool to_reallocate = false;
	    bool over_sized    = false;
    
	    data_slice_p->in_length[enactor_stats->iteration%2][gpu_]
	    = queue_length;
	    if (queue_length == 0) return;
    
	    if (data_slice_p -> keys_in[t][gpu_].GetSize() < queue_length) to_reallocate=true;
	    else {
	        for (i=0;i<num_vertex_associate;i++)
	            if (data_slice_p->vertex_associate_in[t][gpu_][i].GetSize() < queue_length) {to_reallocate=true;break;}
	        if (!to_reallocate)
	            for (i=0;i<num_value__associate;i++)
	                if (data_slice_p->value__associate_in[t][gpu_][i].GetSize() < queue_length) {to_reallocate=true;break;}
	    }
    
	    //if to_reallocate then change GPU index and do cudaMemcpyAsync to copy the data to the new GPU
	    if (to_reallocate)
	    {
	        if (SIZE_CHECK) util::SetDevice(data_slice_p->gpu_idx);
	        // check size for keys
	        if (enactor_stats->retval = Check_Size<SIZE_CHECK, SizeT, VertexId>(
	                                                                            "keys_in", queue_length, &data_slice_p->keys_in[t][gpu_], over_sized,
	                                                                            gpu, enactor_stats->iteration, peer)) return;
        
	        //check size for vertex associates
	        for (i=0;i<num_vertex_associate;i++)
	        {
	            if (enactor_stats->retval = Check_Size<SIZE_CHECK, SizeT, VertexId>(
	                                                                                "vertex_associate_in", queue_length, &data_slice_p->vertex_associate_in[t][gpu_][i], over_sized,
	                                                                                gpu, enactor_stats->iteration, peer)) return;
	            data_slice_p->vertex_associate_ins[t][gpu_][i] = data_slice_p->vertex_associate_in[t][gpu_][i].GetPointer(util::DEVICE);
	        }
        
	        //check size for value associates
	        for (i=0;i<num_value__associate;i++)
	        {
	            if (enactor_stats->retval = Check_Size<SIZE_CHECK, SizeT, Value>(
	                                                                             "value__associate_in", queue_length, &data_slice_p->value__associate_in[t][gpu_][i], over_sized,
	                                                                             gpu, enactor_stats->iteration, peer)) return;
	            data_slice_p->value__associate_ins[t][gpu_][i] = data_slice_p->value__associate_in[t][gpu_][i].GetPointer(util::DEVICE);
	        }
        
	        if (SIZE_CHECK)
	        {
	            if (enactor_stats->retval = data_slice_p->vertex_associate_ins[t][gpu_].Move(util::HOST, util::DEVICE)) return;
	            if (enactor_stats->retval = data_slice_p->value__associate_ins[t][gpu_].Move(util::HOST, util::DEVICE)) return;
	            util::SetDevice(data_slice_l->gpu_idx);
	        }
	    }
    
    
	    if (enactor_stats-> retval = util::GRError(cudaMemcpyAsync(
	                                                               data_slice_p -> keys_in[t][gpu_].GetPointer(util::DEVICE),
	                                                               data_slice_l -> keys_out[peer_].GetPointer(util::DEVICE),
	                                                               sizeof(VertexId) * queue_length, cudaMemcpyDefault, stream),
	                                               "cudaMemcpyPeer keys failed", __FILE__, __LINE__)) return;
    
	    for (int i=0;i<num_vertex_associate;i++)
	    {
	        if (enactor_stats->retval = util::GRError(cudaMemcpyAsync(
	                                                                  data_slice_p->vertex_associate_ins[t][gpu_][i],
	                                                                  data_slice_l->vertex_associate_outs[peer_][i],
	                                                                  sizeof(VertexId) * queue_length, cudaMemcpyDefault, stream),
	                                                  "cudaMemcpyPeer vertex_associate_out failed", __FILE__, __LINE__)) return;
	    }
    
	    for (int i=0;i<num_value__associate;i++)
	    {
	        if (enactor_stats->retval = util::GRError(cudaMemcpyAsync(
	                                                                  data_slice_p->value__associate_ins[t][gpu_][i],
	                                                                  data_slice_l->value__associate_outs[peer_][i],
	                                                                  sizeof(Value) * queue_length, cudaMemcpyDefault, stream),
	                                                  "cudaMemcpyPeer value__associate_out failed", __FILE__, __LINE__)) return;
	    }
	}


	template <typename Problem>
	void ShowDebugInfo(
	                   int           thread_num,
	                   int           peer_,
	                   FrontierAttribute<typename Problem::SizeT>
	                   *frontier_attribute,
	                   EnactorStats *enactor_stats,
	                   typename Problem::DataSlice
	                   *data_slice,
	                   GraphSlice<typename Problem::SizeT, typename Problem::VertexId, typename Problem::Value>
	                   *graph_slice,
	                   util::CtaWorkProgressLifetime
	                   *work_progress,
	                   std::string   check_name = "",
	                   cudaStream_t  stream = 0)
	{
	    typedef typename Problem::SizeT    SizeT;
	    typedef typename Problem::VertexId VertexId;
	    typedef typename Problem::Value    Value;
	    SizeT queue_length;
    
	    //util::cpu_mt::PrintMessage(check_name.c_str(), thread_num, enactor_stats->iteration);
	    //printf("%d \t %d\t \t reset = %d, index = %d\n",thread_num, enactor_stats->iteration, frontier_attribute->queue_reset, frontier_attribute->queue_index);fflush(stdout);
	    //if (frontier_attribute->queue_reset)
	    queue_length = frontier_attribute->queue_length;
	    //else if (enactor_stats->retval = util::GRError(work_progress->GetQueueLength(frontier_attribute->queue_index, queue_length, false, stream), "work_progress failed", __FILE__, __LINE__)) return;
	    //util::cpu_mt::PrintCPUArray<SizeT, SizeT>((check_name+" Queue_Length").c_str(), &(queue_length), 1, thread_num, enactor_stats->iteration);
	    printf("%d\t %lld\t %d\t stage%d\t %s\t Queue_Length = %d\n", thread_num, enactor_stats->iteration, peer_, data_slice->stages[peer_], check_name.c_str(), queue_length);fflush(stdout);
	    //printf("%d \t %d\t \t peer_ = %d, selector = %d, length = %d, p = %p\n",thread_num, enactor_stats->iteration, peer_, frontier_attribute->selector,queue_length,graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE));fflush(stdout);
	    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>((check_name+" keys").c_str(), data_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE), queue_length, thread_num, enactor_stats->iteration,peer_, stream);
	    //if (graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE)!=NULL)
	    //    util::cpu_mt::PrintGPUArray<SizeT, Value   >("valu1", graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE), _queue_length, thread_num, enactor_stats->iteration);
	    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("degrees", data_slice->degrees.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
	    //if (BFSProblem::MARK_PREDECESSORS)
	    //    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("pred1", data_slice[0]->preds.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
	    //if (BFSProblem::ENABLE_IDEMPOTENCE)
	    //    util::cpu_mt::PrintGPUArray<SizeT, unsigned char>("mask1", data_slice[0]->visited_mask.GetPointer(util::DEVICE), (graph_slice->nodes+7)/8, thread_num, enactor_stats->iteration);
	}



	template <typename DataSlice>
	cudaError_t Set_Record(
	                       DataSlice *data_slice,
	                       int iteration,
	                       int peer_,
	                       int stage,
	                       cudaStream_t stream)
	{
	    cudaError_t retval = cudaEventRecord(data_slice->events[iteration%4][peer_][stage],stream);
	    data_slice->events_set[iteration%4][peer_][stage]=true;
	    return retval;
	}



	template <typename DataSlice>
	cudaError_t Check_Record(
	                         DataSlice *data_slice,
	                         int iteration,
	                         int peer_,
	                         int stage_to_check,
	                         int &stage,
	                         bool &to_show)
	{
	    cudaError_t retval = cudaSuccess;
	    to_show = true;
	    if (!data_slice->events_set[iteration%4][peer_][stage_to_check])
	    {
	        to_show = false;
	        stage--;
	    } else {
	        retval = cudaEventQuery(data_slice->events[iteration%4][peer_][stage_to_check]);
	        if (retval == cudaErrorNotReady)
	        {
	            to_show=false;
	            stage--;
	            retval = cudaSuccess;
	        } else if (retval == cudaSuccess)
	        {
	            data_slice->events_set[iteration%4][peer_][stage_to_check]=false;
	        }
	    }
	    return retval;
	}


	#ifdef WITHMPI
	template <
	int      NUM_VERTEX_ASSOCIATES,
	int      NUM_VALUE__ASSOCIATES,
	typename Enactor,
	typename Functor,
	typename Iteration>
	void MPI_Comm_Loop(
	                   ThreadSlice *thread_data)
	{
	    typedef typename Enactor::Problem     Problem   ;
	    typedef typename Problem::SizeT       SizeT     ;
	    typedef typename Problem::VertexId    VertexId  ;
	    typedef typename Problem::Value       Value     ;
	    typedef typename Problem::DataSlice   DataSlice ;
	    Problem      *problem              =  (Problem*) thread_data->problem;
	    Enactor      *enactor              =  (Enactor*) thread_data->enactor;
	    int rank                           =  problem -> mpi_topology -> local_rank;
	    //int num_gpu_local                  =  problem -> mpi_topology -> num_gpus_per_server[rank];
	    int num_gpu_global                 =  problem -> mpi_topology -> total_num_gpus;
	    int gpu                            =  problem -> mpi_topology -> global_gpu_maping[rank][thread_data -> thread_num];
	    struct gunrock::app::MPI_Ring_Buffer<SizeT, VertexId, Value> * mpi_ring_buffer =  problem -> mpi_ring_buffer;
    
    
	    DataSlice    *data_slice           =   problem     -> data_slices        [gpu].GetPointer(util::HOST);
	    /* util::Array1D<SizeT, DataSlice>
	     *s_data_slice         =   problem     -> data_slices; */
	    FrontierAttribute<SizeT>
	    *frontier_attribute   = &(enactor     -> frontier_attribute [gpu * num_gpu_global]);
	    FrontierAttribute<SizeT>
	    *s_frontier_attribute = &(enactor     -> frontier_attribute [0         ]);
	    EnactorStats *enactor_stats        = &(enactor     -> enactor_stats      [gpu * num_gpu_global]);
	    EnactorStats *s_enactor_stats      = &(enactor     -> enactor_stats      [0         ]);
	    util::CtaWorkProgressLifetime
	    *work_progress        = &(enactor     -> work_progress      [gpu * num_gpu_global]);
	    
	    int          *stages               =   data_slice  -> stages .GetPointer(util::HOST);
	    bool         *to_show              =   data_slice  -> to_show.GetPointer(util::HOST);
	    printf("MPI loop ready for action\n");
    
	    //header
	    //3 integers: sending gpu, iteration, length of context
    
	    while(!mpi_ring_buffer->all_done)
	    {
	        int recv_counter = problem->mpi_topology->num_servers-1;
        
	        //for all peers, do a blocking call to wait for the header first and then the data
	        while(recv_counter!=0){
	            int header_length=3;
	            int header[4];
	            MPI_Status status;
				MPI_Request receive_request;

            
	            if(MPI_SUCCESS != MPI_Irecv(&header, header_length, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &receive_request))
	            {
	                fprintf(stderr,"MPI_Recv error in MPI_Comm_Loop (header)\n");
	                return;
	            }
				int received=false;
				while(!mpi_ring_buffer->all_done && !received){
					sleep(0);
					MPI_Test(&receive_request, &received, &status);
				}
				if(mpi_ring_buffer->all_done) return; 
				
	            int sender_rank     = status.MPI_SOURCE;
	            int sender_gpu      = header[0];
	            //int sender_iter     = header[1];
	            int sender_msg_len  = header[2];
	            //int gpu_            = header[3];
	            //header successfully received
	            if(Enactor::DEBUG)
	            {
	                printf("Incoming message from %i: [%i, %i, %i, %i] ",sender_rank, header[0], header[1], header[2], header[3]);
	            }
            
            
	            //int peer_ = sender_gpu<gpu? sender_gpu+1 : sender_gpu;
            
            
	            //check for buffer overflow
	            int incoming_ring_buffer_pos = (mpi_ring_buffer->front_pos[sender_gpu] + 1) % mpi_ring_buffer->length;
	            if(incoming_ring_buffer_pos  == mpi_ring_buffer->back_pos[sender_gpu])
	            {
	                fprintf(stderr,"incoming buffer overflow\n");
	                return;
	            }
            
	            //check if the memory allocated in the ring buffer is large enough
	            if(mpi_ring_buffer->allocated[sender_gpu][incoming_ring_buffer_pos] < sender_msg_len)
	            {
	                //free buffer if previously allocated
	                if(mpi_ring_buffer->allocated[sender_gpu][incoming_ring_buffer_pos]>0)
	                {
	                    free(mpi_ring_buffer->buffer_keys[sender_gpu][incoming_ring_buffer_pos]);
	                    free(mpi_ring_buffer->buffer_vertex[sender_gpu][incoming_ring_buffer_pos]);
	                    free(mpi_ring_buffer->buffer_value[sender_gpu][incoming_ring_buffer_pos]);
	                }
	                //allocate buffer
	                mpi_ring_buffer->buffer_keys[sender_gpu][incoming_ring_buffer_pos]=(VertexId *)malloc(sizeof(VertexId) * sender_msg_len);
	                if(mpi_ring_buffer->buffer_keys[sender_gpu][incoming_ring_buffer_pos]==NULL)
	                {
	                    fprintf(stderr,"Not enough memory. Aborting . %s:%d\n",__FILE__,__LINE__);
	                }
	                mpi_ring_buffer->buffer_vertex[sender_gpu][incoming_ring_buffer_pos]=(VertexId *)malloc(sizeof(VertexId) * sender_msg_len * mpi_ring_buffer->num_vertex_associate);
	                if(mpi_ring_buffer->buffer_vertex[sender_gpu][incoming_ring_buffer_pos]==NULL)
	                {
	                    fprintf(stderr,"Not enough memory. Aborting . %s:%d\n",__FILE__,__LINE__);
	                }
	                mpi_ring_buffer->buffer_value[sender_gpu][incoming_ring_buffer_pos]=(Value *)malloc(sizeof(Value) * sender_msg_len * mpi_ring_buffer->num_value__associate);
	                if(mpi_ring_buffer->buffer_value[sender_gpu][incoming_ring_buffer_pos]==NULL)
	                {
	                    fprintf(stderr,"Not enough memory. Aborting . %s:%d\n",__FILE__,__LINE__);
	                }
	                mpi_ring_buffer->allocated[sender_gpu][incoming_ring_buffer_pos]=sender_msg_len;
	            }
            	mpi_ring_buffer->msg_length[sender_gpu][incoming_ring_buffer_pos]=sender_msg_len;
					
	            if(MPI_SUCCESS != MPI_Recv((&mpi_ring_buffer->buffer_keys[sender_gpu]), sender_msg_len, mpi_ring_buffer->mpi_vertex_type, sender_rank, sender_gpu, MPI_COMM_WORLD, &status))
	            {
	                fprintf(stderr,"MPI_Recv error in MPI_Comm_Loop (header)\n");
	                return;
	            }
	            int offset=0;
	            for(int i=0;i<mpi_ring_buffer->num_vertex_associate; i++){
	                if(MPI_SUCCESS != MPI_Recv(&(mpi_ring_buffer->buffer_vertex[sender_gpu])+offset, sender_msg_len, mpi_ring_buffer->mpi_vertex_type, sender_rank, sender_gpu, MPI_COMM_WORLD, &status))
	                {
	                    fprintf(stderr,"MPI_Recv error in MPI_Comm_Loop (header)\n");
	                    return;
	                }
	                offset+=sender_msg_len;
	            }
	            offset=0;
	            for(int j=0;j<mpi_ring_buffer->num_value__associate; j++){
	                if(MPI_SUCCESS != MPI_Recv(&(mpi_ring_buffer->buffer_value[sender_gpu]), sender_msg_len, mpi_ring_buffer->mpi_value__type, sender_rank, sender_gpu, MPI_COMM_WORLD, &status))
	                {
	                    fprintf(stderr,"MPI_Recv error in MPI_Comm_Loop (header)\n");
	                    return;
	                }
	                offset+=sender_msg_len;
	            }
	            mpi_ring_buffer->front_pos[sender_gpu]=incoming_ring_buffer_pos;
	            recv_counter--;
	        }
	    }
    
	}
	
	template <
	bool     SIZE_CHECK,
	typename SizeT,
	typename VertexId,
	typename Value,
	typename GraphSlice,
	typename DataSlice,
	SizeT    num_vertex_associate,
	SizeT    num_value__associate>
	void MPI_buffer2receiving_device(
		struct gunrock::app::MPI_Ring_Buffer<SizeT,VertexId, Value>
			            *mpi_ring_buffer,
		int              peer_, //virtual peer id
		int              peer,  //real peer id
		gunrock::app::EnactorStats
			            *enactor_stats,
		DataSlice       *target_data_slice,
		cudaStream_t     stream)
	{
		int t  = enactor_stats->iteration%2;
		if(!mpi_ring_buffer->has_data(peer))
		{
			fprintf(stderr,"Unable to access ring buffer data %s:%d\n",__FILE__,__LINE__);
			return;
		}
		int buffer_pos = mpi_ring_buffer->back_pos[peer];
		int queue_length = mpi_ring_buffer->msg_length[peer][buffer_pos];
		
	    if (enactor_stats-> retval = util::GRError(cudaMemcpyAsync(
	                                                               target_data_slice -> keys_in[t][peer_].GetPointer(util::DEVICE),
	                                                               mpi_ring_buffer -> buffer_keys[peer][buffer_pos],
	                                                               sizeof(VertexId) * queue_length, cudaMemcpyDefault, stream),
	                                               "cudaMemcpyPeer keys failed", __FILE__, __LINE__)) return;
    
	    for (int i=0;i<num_vertex_associate;i++)
	    {
	        if (enactor_stats->retval = util::GRError(cudaMemcpyAsync(
	                                                                  target_data_slice->vertex_associate_ins[t][peer_][i],
	                                                                  mpi_ring_buffer -> buffer_vertex[peer][buffer_pos]+i*queue_length,
	                                                                  sizeof(VertexId) * queue_length, cudaMemcpyDefault, stream),
	                                                  "cudaMemcpyPeer vertex_associate_out failed", __FILE__, __LINE__)) return;
	    }
    
	    for (int i=0;i<num_value__associate;i++)
	    {
	        if (enactor_stats->retval = util::GRError(cudaMemcpyAsync(
	                                                                  target_data_slice->value__associate_ins[t][peer_][i],
	                                                                  mpi_ring_buffer -> buffer_value[peer][buffer_pos]+i*queue_length,
	                                                                  sizeof(Value) * queue_length, cudaMemcpyDefault, stream),
	                                                  "cudaMemcpyPeer value__associate_out failed", __FILE__, __LINE__)) return;
	    }
		
	}

	#endif



	template <
	int      NUM_VERTEX_ASSOCIATES,
	int      NUM_VALUE__ASSOCIATES,
	typename Enactor,
	typename Functor,
	typename Iteration>
	void Iteration_Loop(
	                    ThreadSlice *thread_data)
	{
	    typedef typename Enactor::Problem     Problem   ;
	    typedef typename Problem::SizeT       SizeT     ;
	    typedef typename Problem::VertexId    VertexId  ;
	    typedef typename Problem::Value       Value     ;
	    typedef typename Problem::DataSlice   DataSlice ;
	    typedef GraphSlice<SizeT, VertexId, Value>  GraphSlice;
    
    
    
	    Problem      *problem              =  (Problem*) thread_data->problem;
	    Enactor      *enactor              =  (Enactor*) thread_data->enactor;
	#ifdef WITHMPI
	    int rank                           =  problem -> mpi_topology -> local_rank;
	    //int num_gpu_local                  =  problem -> mpi_topology -> num_gpus_per_server[rank];
	    int num_gpu_global                 =  problem -> mpi_topology -> total_num_gpus;
	    int gpu                            =  problem -> mpi_topology -> global_gpu_maping[rank][thread_data -> thread_num];
	    struct gunrock::app::MPI_Ring_Buffer<SizeT, VertexId, Value> * mpi_ring_buffer =  problem -> mpi_ring_buffer;
	#else
	    int rank                           =  0;
	    //int num_gpu_local                  =  problem -> num_gpu;
	    int num_gpu_global                 =  problem -> num_gpu;
	    int gpu                            =  thread_data -> thread_num;
	#endif
    
	    DataSlice    *data_slice           =   problem     -> data_slices        [gpu].GetPointer(util::HOST);
	    util::Array1D<SizeT, DataSlice>
	    *s_data_slice         =   problem     -> data_slices;
	    GraphSlice   *graph_slice          =   problem     -> graph_slices       [gpu] ;
	    GraphSlice   **s_graph_slice       =   problem     -> graph_slices;
	    FrontierAttribute<SizeT>
	    *frontier_attribute   = &(enactor     -> frontier_attribute [gpu * num_gpu_global]);
	    FrontierAttribute<SizeT>
	    *s_frontier_attribute = &(enactor     -> frontier_attribute [0         ]);
	    EnactorStats *enactor_stats        = &(enactor     -> enactor_stats      [gpu * num_gpu_global]);
	    EnactorStats *s_enactor_stats      = &(enactor     -> enactor_stats      [0         ]);
	    util::CtaWorkProgressLifetime
	    *work_progress        = &(enactor     -> work_progress      [gpu * num_gpu_global]);
	    ContextPtr   *context              =   thread_data -> context;
	    int          *stages               =   data_slice  -> stages .GetPointer(util::HOST);
	    bool         *to_show              =   data_slice  -> to_show.GetPointer(util::HOST);
	    cudaStream_t *streams              =   data_slice  -> streams.GetPointer(util::HOST);
	    SizeT         Total_Length         =   0;
	    cudaError_t   tretval              =   cudaSuccess;
	    int           grid_size            =   0;
	    std::string   mssg                 =   "";
	    int           pre_stage            =   0;
	    size_t        offset               =   0;
	    int           iteration            =   0;
	    int           selector             =   0;
	    util::DoubleBuffer<SizeT, VertexId, Value>
	    *frontier_queue_      =   NULL;
	    FrontierAttribute<SizeT>
	    *frontier_attribute_  =   NULL;
	    EnactorStats *enactor_stats_       =   NULL;
	    util::CtaWorkProgressLifetime
	    *work_progress_       =   NULL;
	    util::Array1D<SizeT, SizeT>
	    *scanned_edges_       =   NULL;
	    int           peer, peer_, peer__, gpu_, i, iteration_, wait_count;
	    bool          over_sized;
    
	    printf("Iteration entered\n");fflush(stdout);
    
	#ifdef WITHMPI
	    MPI_Request **sent_requests = (MPI_Request **)malloc(sizeof(MPI_Request *)*num_gpu_global);
	    for(int i=0;i<num_gpu_global; i++)
	    {
	        sent_requests[i]=(MPI_Request *)malloc(sizeof(MPI_Request)*4);
	    }
    
	    while (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, problem -> mpi_topology))
	#else
	        while (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus_local))
	#endif
	        {
	            Total_Length             = 0;
	            data_slice->wait_counter = 0;
	            tretval                  = cudaSuccess;
	            if (num_gpu_global>1 && enactor_stats[0].iteration>0)
	            {
	                frontier_attribute[0].queue_reset  = true;
	                frontier_attribute[0].queue_offset = 0;
	                for (i=1; i<num_gpu_global; i++)
	                {
	                    frontier_attribute[i].selector     = frontier_attribute[0].selector;
	                    frontier_attribute[i].advance_type = frontier_attribute[0].advance_type;
	                    frontier_attribute[i].queue_offset = 0;
	                    frontier_attribute[i].queue_reset  = true;
	                    frontier_attribute[i].queue_index  = frontier_attribute[0].queue_index;
	                    frontier_attribute[i].current_label= frontier_attribute[0].current_label;
	                    enactor_stats     [i].iteration    = enactor_stats     [0].iteration;
	                }
	            }
	            else
	            {
	                frontier_attribute[0].queue_offset = 0;
	                frontier_attribute[0].queue_reset  = true;
	            }
	            for (peer=0; peer<num_gpu_global; peer++)
	            {
	                stages [peer               ] = 0   ;
	                stages [peer+num_gpu_global] = 0   ;
	                to_show[peer               ] = true;
	                to_show[peer+num_gpu_global] = true;
	                for (i=0; i<data_slice->num_stages; i++)
	                    data_slice->events_set[enactor_stats[0].iteration%4][peer][i]=false;
	            }
            
	#ifdef WITHMPI
	            while (data_slice->wait_counter < num_gpu_global*2
	                   && (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, problem->mpi_topology)))
	#else
	                while (data_slice->wait_counter < num_gpu_global*2
	                       && (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus_local)))
	#endif
	                {
	                    for (peer__=0; peer__<num_gpu_global*2; peer__++)
	                    /*
	                     *
	                     *  peer is the id of the peer to communicate with
	                     *  peer_ is the transformed id, because local gpu id has to be 0
	                     *  peer__ is a lazy way of adding another communcation channel:
	                     *  peer_ = 0 local GPU
	                     *  peer_ = 1..num_gpus-1  receiving data channel
	                     *  peer_ = num_gpus       not in use
	                     *  peer_ = num_gpus+1 .. 2*num_gpus-1 sending data
	                     *
	                     *  if local gpu (given by thread_num) is 1 and peer = {0,2,3},
	                     *  then peer_={1,2,3}.
	                     *
	                     * so gpu x sees peer y according to the table below
	                     *
	                     *   gpu      peer-->    0    1    2    3
	                     *  ---------------------------------------
	                     *    0                  0    1    2    3
	                     *    1                  1    0    2    3      <--peer_
	                     *    2                  1    2    0    3
	                     *    3                  1    2    3    0
	                     *
	                     *  gpu_ is the local gpu id from peer_ point of view
	                     *  (simply the transposed matrix) gpu_ will be set
	                     *  to 1 for peer_=1 (which would be GPU0) and gpu_=2 for peer_=2,3
	                     *
	                     *   gpu      peer/gpu_-->     0      1      2      3
	                     *  ----------------------------------------------------
	                     *    0                       0/x    1/1    2/1    3/1
	                     *    1                       1/1    0/x    2/2    3/2      <--peer_
	                     *    2                       1/2    2/2    0/x    3/3
	                     *    3                       1/3    2/3    3/3    0/x
	                     *
	                     *   because GPU3 sees GPU1 as peer_=2, whereas
	                     *   GPU1 sees GPU3 as peer_=3. hence for gpu=3 and peer=1,
	                     *   peer_=2 and gpu_=3
	                     *
	                     */
	                    {
	                        peer_               = (peer__%num_gpu_global);
	                        peer                = peer_<= gpu? peer_-1   : peer_       ;
	                        gpu_                = peer <  gpu? gpu: gpu+1;
	                        iteration           = enactor_stats[peer_].iteration;
	                        iteration_          = iteration%4;
	                        pre_stage           = stages[peer__];
	                        selector            = frontier_attribute[peer_].selector;
	                        frontier_queue_     = &(data_slice->frontier_queues[peer_]);
	                        scanned_edges_      = &(data_slice->scanned_edges  [peer_]);
	                        frontier_attribute_ = &(frontier_attribute         [peer_]);
	                        enactor_stats_      = &(enactor_stats              [peer_]);
	                        work_progress_      = &(work_progress              [peer_]);
                        
	                        if (Enactor::DEBUG && to_show[peer__])
	                        {
	                            mssg=" ";mssg[0]='0'+data_slice->wait_counter;
	                            ShowDebugInfo<Problem>(
	                                                   gpu,
	                                                   peer__,
	                                                   frontier_attribute_,
	                                                   enactor_stats_,
	                                                   data_slice,
	                                                   graph_slice,
	                                                   work_progress_,
	                                                   mssg,
	                                                   streams[peer__]);
	                        }
	                        to_show[peer__]=true;
                        
	                        switch (stages[peer__])
	                        {
	                            case 0: // Assign marker & Scan and send & receive
	#ifdef WITHMPI
	                                if (peer_==0) {
	                                    if (peer__==num_gpu_global	|| frontier_attribute_->queue_length==0)
	                                    {
	                                        stages[peer__]=3;
	                                    } else if (!Iteration::HAS_SUBQ) { //process subqueue if that is necessary
	                                        stages[peer__]=2;
	                                    }
	                                    break;
	                                } else if ((iteration==0 || data_slice->out_length[peer_]==0) && peer__>num_gpu_global) {
	                                    //initialize the income stream event flag
	                                    Set_Record(data_slice, iteration, peer_, 0, streams[peer__]);
	                                    stages[peer__]=3;
	                                    break;
	                                }
                                
	                                if (peer__<num_gpu_global) //incoming data
	                                { //wait and expand incoming
	                                    if(problem->mpi_topology->rank_of_gpu[peer_] != problem->mpi_topology->rank_of_gpu[gpu])  //if data was received via MPI from another node
	                                    {
											
	                                        if(!mpi_ring_buffer->has_data(peer_))
	                                        {//if not, do nothing
	                                            to_show[peer__]=false;
	                                            stages[peer__]--;        //at the end ofthe loop, the stages increase. Hence "--" means to stay in stage 0
	                                            break;
	                                        }
	                                        if(!mpi_ring_buffer->allocated[peer_][0]) //check if actually data is in the input queue
	                                        {
	                                            stages[peer__]=3;
	                                            break;
	                                        }
	                                        offset = 0;
	                                        //copy data to the expand_incoming_array from
	                                        //    vertex_associate_ins[iteration%2][peer_]
	                                        //    value__associate_ins[iteration%2][peer_]
	                                        //    vertex_associate_orgs
	                                        //    value__associate_orgs
	                                        //
	                                        int buffer_pos = mpi_ring_buffer->back_pos[peer];
                                        	int msg_length = mpi_ring_buffer->msg_length[peer][buffer_pos];
											
											/* copy data from buffer_1 to buffer_2
											MPI_buffer2receiving_device(mpi_ring_buffer,peer,
											s_data_slice  [gpu].GetPointer(util::HOST),streams[peer__]);
											*/
											MPI_buffer2receiving_device<
	                                        Enactor::SIZE_CHECK,
	                                        SizeT,
	                                        VertexId,
	                                        Value,
	                                        GraphSlice,
	                                        DataSlice,
	                                        NUM_VERTEX_ASSOCIATES,
	                                        NUM_VALUE__ASSOCIATES
	                                        >(
												mpi_ring_buffer,
												peer_,   //virtual peer number
												peer,    //real peer number
												enactor_stats,
												s_data_slice[gpu].GetPointer(util::HOST),
												streams[peer__]);
											
											//wait for copy to complete. This should be asynchroneously done
								            if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(data_slice->streams[peer__]),
								                        "cudaStreamSynchronize failed", __FILE__, __LINE__)) break;
											
	                                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
	                                               data_slice -> vertex_associate_ins[iteration%2][peer_].GetPointer(util::HOST),
	                                               sizeof(SizeT*   ) * NUM_VERTEX_ASSOCIATES);
	                                        offset += sizeof(SizeT*   ) * NUM_VERTEX_ASSOCIATES ;
	                                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
	                                               data_slice -> value__associate_ins[iteration%2][peer_].GetPointer(util::HOST),
	                                               sizeof(VertexId*) * NUM_VALUE__ASSOCIATES);
	                                        offset += sizeof(VertexId*) * NUM_VALUE__ASSOCIATES ;
	                                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
	                                               data_slice -> vertex_associate_orgs.GetPointer(util::HOST),
	                                               sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
	                                        offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
	                                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
	                                               data_slice -> value__associate_orgs.GetPointer(util::HOST),
	                                               sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
	                                        offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;

	                                    }
	                                    else   //data was received via CudaMemcpy
	                                    {
	                                        if (!(s_data_slice[peer]->events_set[iteration_][gpu_][0])) //check if event happened
	                                        {   //if not, do nothing
	                                            to_show[peer__]=false;
	                                            stages[peer__]--;        //at the end ofthe loop, the stages increase. Hence "--" means to stay in stage 0
	                                            break;
	                                        }
                                        
	                                        //reset event flag
	                                        s_data_slice[peer]->events_set[iteration_][gpu_][0]=false;
	                                        frontier_attribute_->queue_length = data_slice->in_length[iteration%2][peer_];
	                                        data_slice->in_length[iteration%2][peer_]=0;
	                                        if (frontier_attribute_->queue_length ==0)  //check if actually data is in the input queue
	                                        {
	                                            stages[peer__]=3;
	                                            break;
	                                        }
                                        
	                                        offset = 0;
	                                        //
	                                        //copy data to the expand_incoming_array from
	                                        //    vertex_associate_ins[iteration%2][peer_]
	                                        //    value__associate_ins[iteration%2][peer_]
	                                        //    vertex_associate_orgs
	                                        //    value__associate_orgs
	                                        //
	                                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
	                                               data_slice -> vertex_associate_ins[iteration%2][peer_].GetPointer(util::HOST),
	                                               sizeof(SizeT*   ) * NUM_VERTEX_ASSOCIATES);
	                                        offset += sizeof(SizeT*   ) * NUM_VERTEX_ASSOCIATES ;
	                                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
	                                               data_slice -> value__associate_ins[iteration%2][peer_].GetPointer(util::HOST),
	                                               sizeof(VertexId*) * NUM_VALUE__ASSOCIATES);
	                                        offset += sizeof(VertexId*) * NUM_VALUE__ASSOCIATES ;
	                                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
	                                               data_slice -> vertex_associate_orgs.GetPointer(util::HOST),
	                                               sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
	                                        offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
	                                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
	                                               data_slice -> value__associate_orgs.GetPointer(util::HOST),
	                                               sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
	                                        offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;
	                                    }
	                                    //copy from host to GPU
	                                    data_slice->expand_incoming_array[peer_].Move(util::HOST, util::DEVICE, offset, 0, streams[peer_]);
                                    
	                                    grid_size = frontier_attribute_->queue_length/256+1;
	                                    if (grid_size>512) grid_size=512;
                                    
	                                    //wait until copy to GPU is completed
	                                    cudaStreamWaitEvent(streams[peer_],
	                                                        s_data_slice[peer]->events[iteration_][gpu_][0], 0);
	                                    //unpack data
	                                    Iteration::template Expand_Incoming<NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
	                                                                                                                       grid_size, 256,
	                                                                                                                       offset,
	                                                                                                                       streams[peer_],
	                                                                                                                       frontier_attribute_->queue_length,
	                                                                                                                       data_slice ->keys_in[iteration%2][peer_].GetPointer(util::DEVICE),
	                                                                                                                       &frontier_queue_->keys[selector^1],
	                                                                                                                       offset,
	                                                                                                                       data_slice ->expand_incoming_array[peer_].GetPointer(util::DEVICE),
	                                                                                                                       data_slice);
	                                    frontier_attribute_->selector^=1;
	                                    frontier_attribute_->queue_index++;
	                                    //process subqueue if that is needed
	                                    if (!Iteration::HAS_SUBQ) {
	                                        Set_Record(data_slice, iteration, peer_, 2, streams[peer__]);
	                                        stages[peer__]=2;
	                                    }
	                                }
	                                else
	                                {
										
	                                    if(problem->mpi_topology->rank_of_gpu[peer_] != problem->mpi_topology->rank_of_gpu[gpu])  //if data has to be sent via MPI to another node
	                                    {
	                                        //MPI_PUSH_NEIGHBOR
	                                        MPI_PushNeighbor <
	                                        Enactor::SIZE_CHECK,
	                                        SizeT,
	                                        VertexId,
	                                        Value,
	                                        GraphSlice,
	                                        DataSlice,
	                                        NUM_VERTEX_ASSOCIATES,
	                                        NUM_VALUE__ASSOCIATES
	                                        > (gpu,
	                                           peer_,
	                                           iteration,
	                                           problem->mpi_topology->rank_of_gpu[peer_],
	                                           data_slice->out_length[peer_],
	                                           enactor_stats_,
	                                           s_data_slice  [gpu] .GetPointer(util::HOST),
	                                           s_data_slice  [peer].GetPointer(util::HOST),
	                                           s_graph_slice [gpu],
	                                           s_graph_slice [peer],
	                                           sent_requests,
	                                           mpi_ring_buffer->mpi_vertex_type,
	                                           mpi_ring_buffer->mpi_value__type
	                                           );
	                                    }
	                                    else //copy data on same node
	                                    {
	                                        PushNeighbor <
	                                        Enactor::SIZE_CHECK,
	                                        SizeT,
	                                        VertexId,
	                                        Value,
	                                        GraphSlice,
	                                        DataSlice,
	                                        NUM_VERTEX_ASSOCIATES,
	                                        NUM_VALUE__ASSOCIATES
	                                        > (
	                                           gpu,
	                                           peer,
	                                           data_slice->out_length[peer_],
	                                           enactor_stats_,
	                                           s_data_slice  [gpu].GetPointer(util::HOST),
	                                           s_data_slice  [peer]      .GetPointer(util::HOST),
	                                           s_graph_slice [gpu],
	                                           s_graph_slice [peer],
	                                           streams       [peer__]
	                                           );
	                                        //set event that data has been copied.
	                                        // the following line means set s_data_slice[gpu]->set_event[iteration%4][peer_][0]=true
	                                        Set_Record(data_slice, iteration, peer_, stages[peer__], streams[peer__]);
	                                    }
	                                    stages[peer__]=3;
	                                }
                                
	                                break;
	#else
	                                if (peer_==0) {
	                                    if (peer__==num_gpu_global || frontier_attribute_->queue_length==0)
	                                    {
	                                        stages[peer__]=3;
	                                    } else if (!Iteration::HAS_SUBQ) { //process subqueue if that is necessary
	                                        stages[peer__]=2;
	                                    }
	                                    break;
	                                } else if ((iteration==0 || data_slice->out_length[peer_]==0) && peer__>num_gpu_global) {
	                                    //initialize the income stream event flag
	                                    Set_Record(data_slice, iteration, peer_, 0, streams[peer__]);
	                                    stages[peer__]=3;
	                                    break;
	                                }
                                
	                                if (peer__<num_gpu_global) //incoming data
	                                { //wait and expand incoming
	                                    if (!(s_data_slice[peer]->events_set[iteration_][gpu_][0])) //check if event happened
	                                    {   //if not, do nothing
	                                        to_show[peer__]=false;
	                                        stages[peer__]--;        //at the end ofthe loop, the stages increase. Hence "--" means to stay in stage 0
	                                        break;
	                                    }
                                    
	                                    //reset event flag
	                                    s_data_slice[peer]->events_set[iteration_][gpu_][0]=false;
	                                    frontier_attribute_->queue_length = data_slice->in_length[iteration%2][peer_];
	                                    data_slice->in_length[iteration%2][peer_]=0;
	                                    if (frontier_attribute_->queue_length ==0)  //check if actually data is in the input queue
	                                    {   stages[peer__]=3;break;}
                                    
	                                    offset = 0;
	                                    //copy 1 element of data to the expand_incoming_array from
	                                    //    vertex_associate_ins[iteration%2][peer_]
	                                    //    value__associate_ins[iteration%2][peer_]
	                                    //    vertex_associate_orgs
	                                    //    value__associate_orgs
	                                    //
	                                    memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
	                                           data_slice -> vertex_associate_ins[iteration%2][peer_].GetPointer(util::HOST),
	                                           sizeof(SizeT*   ) * NUM_VERTEX_ASSOCIATES);
	                                    offset += sizeof(SizeT*   ) * NUM_VERTEX_ASSOCIATES ;
	                                    memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
	                                           data_slice -> value__associate_ins[iteration%2][peer_].GetPointer(util::HOST),
	                                           sizeof(VertexId*) * NUM_VALUE__ASSOCIATES);
	                                    offset += sizeof(VertexId*) * NUM_VALUE__ASSOCIATES ;
	                                    memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
	                                           data_slice -> vertex_associate_orgs.GetPointer(util::HOST),
	                                           sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
	                                    offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
	                                    memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
	                                           data_slice -> value__associate_orgs.GetPointer(util::HOST),
	                                           sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
	                                    offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;
                                    
	                                    //copy from host to GPU
	                                    data_slice->expand_incoming_array[peer_].Move(util::HOST, util::DEVICE, offset, 0, streams[peer_]);
                                    
	                                    grid_size = frontier_attribute_->queue_length/256+1;
	                                    if (grid_size>512) grid_size=512;
                                    
	                                    //wait until copy to GPU is completed
	                                    cudaStreamWaitEvent(streams[peer_],
	                                                        s_data_slice[peer]->events[iteration_][gpu_][0], 0);
	                                    //unpack data
	                                    Iteration::template Expand_Incoming<NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
	                                                                                                                       grid_size, 256,
	                                                                                                                       offset,
	                                                                                                                       streams[peer_],
	                                                                                                                       frontier_attribute_->queue_length,
	                                                                                                                       data_slice ->keys_in[iteration%2][peer_].GetPointer(util::DEVICE),
	                                                                                                                       &frontier_queue_->keys[selector^1],
	                                                                                                                       offset,
	                                                                                                                       data_slice ->expand_incoming_array[peer_].GetPointer(util::DEVICE),
	                                                                                                                       data_slice);
	                                    frontier_attribute_->selector^=1;
	                                    frontier_attribute_->queue_index++;
	                                    //process subqueue if that is needed
	                                    if (!Iteration::HAS_SUBQ) {
	                                        Set_Record(data_slice, iteration, peer_, 2, streams[peer__]);
	                                        stages[peer__]=2;
	                                    }
	                                }
	                                else
	                                { //peer__ = num_gpu .. 2*num_gpu-1 --> sending data channel
	                                    //copy data to different GPU on same node
	                                    PushNeighbor <Enactor::SIZE_CHECK,SizeT,
	                                    VertexId, Value, GraphSlice, DataSlice,
	                                    NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
	                                                                                   gpu,
	                                                                                   peer,
	                                                                                   data_slice->out_length[peer_],
	                                                                                   enactor_stats_,
	                                                                                   s_data_slice  [gpu].GetPointer(util::HOST),
	                                                                                   s_data_slice  [peer]      .GetPointer(util::HOST),
	                                                                                   s_graph_slice [gpu], s_graph_slice [peer],
	                                                                                   streams       [peer__]);
	                                    //set event that data has been copied.
	                                    // the following line means set s_data_slice[gpu]->set_event[iteration%4][peer_][0]=true
	                                    Set_Record(data_slice, iteration, peer_, stages[peer__], streams[peer__]);
	                                    stages[peer__]=3;
	                                }
	                                break;
	#endif
                                
	                            case 1: //Compute Output Length
	                                if (enactor_stats_->retval = Iteration::Compute_OutputLength(
	                                                                                             frontier_attribute_,
	                                                                                             graph_slice    ->row_offsets     .GetPointer(util::DEVICE),
	                                                                                             graph_slice    ->column_indices  .GetPointer(util::DEVICE),
	                                                                                             frontier_queue_->keys[selector]  .GetPointer(util::DEVICE),
	                                                                                             scanned_edges_,
	                                                                                             graph_slice    ->nodes,
	                                                                                             graph_slice    ->edges,
	                                                                                             context          [peer_][0],
	                                                                                             streams          [peer_],
	                                                                                             gunrock::oprtr::advance::V2V, true)) break;
                                
	                                frontier_attribute_->output_length.Move(util::DEVICE, util::HOST,1,0,streams[peer_]);
	                                if (Enactor::SIZE_CHECK)
	                                {
	                                    Set_Record(data_slice, iteration, peer_, stages[peer_], streams[peer_]);
	                                }
	                                break;
                                
	                            case 2: //SubQueue Core
	                                //in case there needs to be some preprocessing on the individuals sub-queues before the queues are merged into a new frontier
	                                if (Enactor::SIZE_CHECK)
	                                {
	                                    if (enactor_stats_ -> retval = Check_Record (                        //check if there is data to process
	                                                                                 data_slice, iteration, peer_,
	                                                                                 stages[peer_]-1, stages[peer_], to_show[peer_])) break;
	                                    if (to_show[peer_]==false) break;
	                                    Iteration::Check_Queue_Size(
	                                                                gpu,
	                                                                peer_,
	                                                                frontier_attribute_->output_length[0] + 2,
	                                                                frontier_queue_,
	                                                                frontier_attribute_,
	                                                                enactor_stats_,
	                                                                graph_slice);
	                                }
	                                //do the subqueue processing
	                                Iteration::SubQueue_Core(
	                                                         gpu,
	                                                         peer_,
	                                                         frontier_queue_,
	                                                         scanned_edges_,
	                                                         frontier_attribute_,
	                                                         enactor_stats_,
	                                                         data_slice,
	                                                         s_data_slice[gpu].GetPointer(util::DEVICE),
	                                                         graph_slice,
	                                                         &(work_progress[peer_]),
	                                                         context[peer_],
	                                                         streams[peer_]);
                                
                                
	                                if (enactor_stats_->retval = work_progress[peer_].GetQueueLength(
	                                                                                                 frontier_attribute_->queue_index,
	                                                                                                 frontier_attribute_->queue_length,
	                                                                                                 false,
	                                                                                                 streams[peer_],
	                                                                                                 true)) break;
	                                if (num_gpu_global>1)
	                                    Set_Record(data_slice, iteration, peer_, stages[peer_], streams[peer_]);  //set the flag that the data has been processed
	                                break;
                                
	                            case 3: //Copy the subqueues to one common place
	                                if (num_gpu_global <=1)
	                                {
	                                    if (enactor_stats_-> retval = util::GRError(cudaStreamSynchronize(streams[peer_]), "cudaStreamSynchronize failed",__FILE__, __LINE__)) break;
	                                    Total_Length = frontier_attribute_->queue_length;
	                                    to_show[peer_]=false;break;
	                                }
	                                if (Iteration::HAS_SUBQ || peer_!=0) {             //make sure subqueue processing is done
	                                    if (enactor_stats_-> retval = Check_Record(
	                                                                               data_slice, iteration, peer_,
	                                                                               stages[peer_]-1, stages[peer_], to_show[peer_])) break;
	                                    if (to_show[peer_] == false) break;
	                                }
                                
	                                if (!Enactor::SIZE_CHECK)      //if size check is disabled
	                                {
	                                    if (Iteration::HAS_SUBQ)
	                                    {
	                                        if (enactor_stats_->retval =
	                                            Check_Size<false, SizeT, VertexId> ("queue3", frontier_attribute_->output_length[0]+2, &frontier_queue_->keys  [selector^1], over_sized, gpu, iteration, peer_, false)) break;
	                                    }
	                                    if (frontier_attribute_->queue_length ==0) break;
                                    
	                                    if (enactor_stats_->retval =
	                                        Check_Size<false, SizeT, VertexId> ("total_queue", Total_Length + frontier_attribute_->queue_length, &data_slice->frontier_queues[num_gpu_global].keys[0], over_sized, gpu, iteration, peer_, false)) break;
                                    
	                                    //launch copy kernel to copy a subqueue from frontier queue associated with peer_ (a.k.a. frontier_queue_) to one common place, the end of the data in frontier_queues[num_gpus_global]
	                                    //  frontier_queue_->keys[selector].GetPointer(util::DEVICE)
	                                    //     to
	                                    //  data_slice->frontier_queues[num_gpus_global].keys[0].GetPointer(util::DEVICE) + Total_Length
                                    
	                                    util::MemsetCopyVectorKernel<<<256,256, 0, streams[peer_]>>>(
	                                                                                                 data_slice->frontier_queues[num_gpu_global].keys[0].GetPointer(util::DEVICE) + Total_Length,
	                                                                                                 frontier_queue_->keys[selector].GetPointer(util::DEVICE),
	                                                                                                 frontier_attribute_->queue_length);
	                                    if (Problem::USE_DOUBLE_BUFFER)
	                                        util::MemsetCopyVectorKernel<<<256,256,0,streams[peer_]>>>(
	                                                                                                   data_slice->frontier_queues[num_gpu_global].values[0].GetPointer(util::DEVICE) + Total_Length,
	                                                                                                   frontier_queue_->values[selector].GetPointer(util::DEVICE),
	                                                                                                   frontier_attribute_->queue_length);
	                                }
                                
	                                Total_Length += frontier_attribute_->queue_length; //update the total length, used as offset for the new copy position
	                                break;
                                
	                            case 4: //End
	                                data_slice->wait_counter++;
	                                to_show[peer__]=false;
	                                break;
	                            default:
	                                stages[peer__]--;
	                                to_show[peer__]=false;
	                        }
                        
	                        if (Enactor::DEBUG && !enactor_stats_->retval)
	                        {
	                            mssg="stage 0 @ gpu 0, peer_ 0 failed";
	                            mssg[6]=char(pre_stage+'0');
	                            mssg[14]=char(gpu+'0');
	                            mssg[23]=char(peer__+'0');
	                            if (enactor_stats_->retval = util::GRError(mssg, __FILE__, __LINE__)) break;
	                        }
	                        stages[peer__]++;
	                        if (enactor_stats_->retval) break;
	                    }
	                }
	#ifdef WITHMPI
	            if (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, problem->mpi_topology))
	#else
	                if (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus_local))
	#endif
	                {
	                    for (peer_=0;peer_<num_gpu_global*2;peer_++)
	                        data_slice->wait_marker[peer_]=0;
	                    wait_count=0;
	#ifdef WITHMPI
	                    while (wait_count<num_gpu_global*2-1 &&
	                           !Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, problem->mpi_topology))
	#else
	                        while (wait_count<num_gpu_global*2-1 &&
	                               !Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus_local))
	#endif
	                        {
	                            for (peer_=0;peer_<num_gpu_global*2;peer_++)
	                            {
	                                if (peer_==num_gpu_global || data_slice->wait_marker[peer_]!=0)  //peer_==num_gpus (channel to itself, i.e. not needed)
	                                    continue;
	#ifdef WITHMPI
	                                if((problem->mpi_topology->rank_of_gpu[peer_] != problem->mpi_topology->rank_of_gpu[gpu]))                  //check if MPI communication
	                                {
	                                    if (MPI_Communication_Check(sent_requests, peer_, num_gpu_global,1+NUM_VERTEX_ASSOCIATES+NUM_VALUE__ASSOCIATES))   //check if MPI_Sent or receive for peer_ is completed
	                                    {
	                                        data_slice->wait_marker[peer_]=1;
	                                        wait_count++;
	                                        continue;
	                                    }
	                                }
	                                else
	                                {
	                                    tretval = cudaStreamQuery(streams[peer_]);                 //check if all operations in stream for peer_ are completed
	                                    if (tretval == cudaSuccess)
	                                    {
	                                        data_slice->wait_marker[peer_]=1;
	                                        wait_count++;
	                                        continue;
	                                    } else if (tretval != cudaErrorNotReady)
	                                    {
	                                        enactor_stats[peer_%num_gpu_global].retval = tretval;
	                                        break;
	                                    }
	                                }
                                
	#else
	                                tretval = cudaStreamQuery(streams[peer_]);                 //check if all operations in stream for peer_ are completed
	                                if (tretval == cudaSuccess)
	                                {
	                                    data_slice->wait_marker[peer_]=1;
	                                    wait_count++;
	                                    continue;
	                                } else if (tretval != cudaErrorNotReady)
	                                {
	                                    enactor_stats[peer_%num_gpu_global].retval = tretval;
	                                    break;
	                                }
	#endif
	                            }
	                        }
                    
	                    if (Enactor::DEBUG) {printf("%d\t %lld\t \t Subqueue finished. Total_Length= %d\n", gpu, enactor_stats[0].iteration, Total_Length);fflush(stdout);}
	                    grid_size = Total_Length/256+1;
	                    if (grid_size > 512) grid_size = 512;
                    
	                    if (Enactor::SIZE_CHECK)
	                    {
	                        if (enactor_stats[0]. retval =
	                            Check_Size<true, SizeT, VertexId> ("total_queue", Total_Length, &data_slice->frontier_queues[0].keys[frontier_attribute[0].selector], over_sized, gpu, iteration, num_gpu_global, true)) break;
	                        if (Problem::USE_DOUBLE_BUFFER)
	                            if (enactor_stats[0].retval =
	                                Check_Size<true, SizeT, Value> ("total_queue", Total_Length, &data_slice->frontier_queues[0].values[frontier_attribute[0].selector], over_sized, gpu, iteration, num_gpu_global, true)) break;
                        
	                        offset=frontier_attribute[0].queue_length;
	                        //copy queues together to position
	                        // data_slice->frontier_queues[0]
	                        for (peer_=1;peer_<num_gpu_global;peer_++)
	                            if (frontier_attribute[peer_].queue_length !=0) {
	                                util::MemsetCopyVectorKernel<<<256,256, 0, streams[0]>>>(
	                                                                                         data_slice->frontier_queues[0    ].keys[frontier_attribute[0    ].selector].GetPointer(util::DEVICE) + offset,
	                                                                                         data_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
	                                                                                         frontier_attribute[peer_].queue_length);
	                                if (Problem::USE_DOUBLE_BUFFER)
	                                    util::MemsetCopyVectorKernel<<<256,256,0,streams[0]>>>(
	                                                                                           data_slice->frontier_queues[0       ].values[frontier_attribute[0    ].selector].GetPointer(util::DEVICE) + offset,
	                                                                                           data_slice->frontier_queues[peer_   ].values[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
	                                                                                           frontier_attribute[peer_].queue_length);
	                                offset+=frontier_attribute[peer_].queue_length;
	                            }
	                    }
	                    frontier_attribute[0].queue_length = Total_Length;
	                    if (!Enactor::SIZE_CHECK) frontier_attribute[0].selector = 0;
	                    //select frontier queue position
	                    // if size_check is enableled, take data from frontier_attribute[0]
	                    //otherwise take the data from frontier_attribute[num_gpus], where it was copied to in stage 3
	                    frontier_queue_ = &(data_slice->frontier_queues[(Enactor::SIZE_CHECK || num_gpu_global == 1)?0:num_gpu_global]);
	                    if (Iteration::HAS_FULLQ)
	                    {
	                        peer_               = 0;
	                        frontier_queue_     = &(data_slice->frontier_queues[(Enactor::SIZE_CHECK || num_gpu_global==1)?0:num_gpu_global]);
	                        scanned_edges_      = &(data_slice->scanned_edges  [(Enactor::SIZE_CHECK || num_gpu_global==1)?0:num_gpu_global]);
	                        frontier_attribute_ = &(frontier_attribute[peer_]);
	                        enactor_stats_      = &(enactor_stats[peer_]);
	                        work_progress_      = &(work_progress[peer_]);
	                        iteration           = enactor_stats[peer_].iteration;
	                        frontier_attribute_->queue_offset = 0;
	                        frontier_attribute_->queue_reset  = true;
	                        if (!Enactor::SIZE_CHECK) frontier_attribute_->selector     = 0;
                        
	                        Iteration::FullQueue_Gather(
	                                                    gpu,
	                                                    peer_,
	                                                    frontier_queue_,
	                                                    scanned_edges_,
	                                                    frontier_attribute_,
	                                                    enactor_stats_,
	                                                    data_slice,
	                                                    s_data_slice[gpu].GetPointer(util::DEVICE),
	                                                    graph_slice,
	                                                    work_progress_,
	                                                    context[peer_],
	                                                    streams[peer_]);
	                        selector            = frontier_attribute[peer_].selector;
	                        if (enactor_stats_->retval) break;
                        
	                        if (frontier_attribute_->queue_length !=0)
	                        {
	                            if (Enactor::DEBUG) {
	                                mssg = "";
	                                ShowDebugInfo<Problem>(
	                                                       gpu,
	                                                       peer_,
	                                                       frontier_attribute_,
	                                                       enactor_stats_,
	                                                       data_slice,
	                                                       graph_slice,
	                                                       work_progress_,
	                                                       mssg,
	                                                       streams[peer_]);
	                            }
                            
	                            enactor_stats_->retval = Iteration::Compute_OutputLength(
	                                                                                     frontier_attribute_,
	                                                                                     graph_slice    ->row_offsets     .GetPointer(util::DEVICE),
	                                                                                     graph_slice    ->column_indices  .GetPointer(util::DEVICE),
	                                                                                     frontier_queue_->keys[selector].GetPointer(util::DEVICE),
	                                                                                     scanned_edges_,
	                                                                                     graph_slice    ->nodes,
	                                                                                     graph_slice    ->edges,
	                                                                                     context          [peer_][0],
	                                                                                     streams          [peer_],
	                                                                                     gunrock::oprtr::advance::V2V, true);
	                            if (enactor_stats_->retval) break;
                            
	                            frontier_attribute_->output_length.Move(util::DEVICE, util::HOST, 1, 0, streams[peer_]);
	                            if (Enactor::SIZE_CHECK)
	                            {
	                                tretval = cudaStreamSynchronize(streams[peer_]);
	                                if (tretval != cudaSuccess) {enactor_stats_->retval=tretval;break;}
                                
	                                Iteration::Check_Queue_Size(
	                                                            gpu,
	                                                            peer_,
	                                                            frontier_attribute_->output_length[0] + 2,
	                                                            frontier_queue_,
	                                                            frontier_attribute_,
	                                                            enactor_stats_,
	                                                            graph_slice);
                                
	                            }
                            
	                            Iteration::FullQueue_Core(
	                                                      gpu,
	                                                      peer_,
	                                                      frontier_queue_,
	                                                      scanned_edges_,
	                                                      frontier_attribute_,
	                                                      enactor_stats_,
	                                                      data_slice,
	                                                      s_data_slice[gpu].GetPointer(util::DEVICE),
	                                                      graph_slice,
	                                                      work_progress_,
	                                                      context[peer_],
	                                                      streams[peer_]);
	                            if (enactor_stats_->retval) break;
	                            if (!Enactor::SIZE_CHECK)
	                            {
	                                if (enactor_stats_->retval =
	                                    Check_Size<false, SizeT, VertexId> ("queue3", frontier_attribute->output_length[0]+2, &frontier_queue_->keys[selector^1], over_sized, gpu, iteration, peer_, false)) break;
	                            }
	                            selector = frontier_attribute[peer_].selector;
	                            Total_Length = frontier_attribute[peer_].queue_length;
	                        } else {
	                            Total_Length = 0;
	                            for (peer__=0;peer__<num_gpu_global;peer__++)
	                                data_slice->out_length[peer__]=0;
	                        }
	                        if (Enactor::DEBUG) {printf("%d\t %lld\t \t Fullqueue finished. Total_Length= %d\n", gpu, enactor_stats[0].iteration, Total_Length);fflush(stdout);}
	                        frontier_queue_ = &(data_slice->frontier_queues[Enactor::SIZE_CHECK?0:num_gpu_global]);
	                        if (num_gpu_global==1) data_slice->out_length[0]=Total_Length;
	                    }
                    
	                    if (num_gpu_global > 1)
	                    {
	                        Iteration::Iteration_Update_Preds(
	                                                          graph_slice,
	                                                          data_slice,
	                                                          &frontier_attribute[0],
	                                                          &data_slice->frontier_queues[Enactor::SIZE_CHECK?0:num_gpu_global],
	                                                          Total_Length,
	                                                          streams[0]);
	                        Iteration::template Make_Output <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
	                                                                                                        gpu,
	                                                                                                        Total_Length,
	                                                                                                        num_gpu_global,
	                                                                                                        &data_slice->frontier_queues[Enactor::SIZE_CHECK?0:num_gpu_global],
	                                                                                                        &data_slice->scanned_edges[0],
	                                                                                                        &frontier_attribute[0],
	                                                                                                        enactor_stats,
	                                                                                                        &problem->data_slices[gpu],
	                                                                                                        graph_slice,
	                                                                                                        &work_progress[0],
	                                                                                                        context[0],
	                                                                                                        streams[0]);
	                    } else data_slice->out_length[0]= Total_Length;
                    
	                    for (peer_=0;peer_<num_gpu_global;peer_++)
	                        frontier_attribute[peer_].queue_length = data_slice->out_length[peer_];
	                }
	            Iteration::Iteration_Change(enactor_stats->iteration);
	        }
#ifdef WITHMPI
		    for(int i=0;i<num_gpu_global; i++)
		    {
		        free(sent_requests[i]);
		    }
			free(sent_requests);
#endif
	}


	#ifdef WITHMPI

	std::vector<std::string> tokenize(std::string str){
	    std::stringstream strstr(str);
    
	    // use stream iterators to copy the stream to the vector as whitespace separated strings
	    std::istream_iterator<std::string> it(strstr);
	    std::istream_iterator<std::string> end;
	    std::vector<std::string> results(it, end);
	    return results;
	}


	std::vector<std::string> tokenize(std::string str, char splitter){
	    std::stringstream strstr(str);
	    std::string elem;
	    std::vector<std::string> results;
	    while (getline(strstr, elem, splitter)) {
	        results.push_back(elem);
	    }
	    return results;
	}


	/*
	 *
	 *  @brief A reader for the gpu topology config file, in case multiple GPUs across various servers are used
	 *
	 * @param[in] filename The filename of the config file
	 *
	 * \return GPU_Topology
	 *
	 * struct GPU_Topology
	 *	int num_servers                    ;   // the number of involved servers
	 *	int *num_gpus_per_server           ;   // how many GPUs are used in each server
	 *	int ** global_gpu_maping           ;   // global_gpu_maping[server][thread] gives a globally unique number (for data splitting)
	 *	int ** local_gpu_mapping           ;   // local_gpu_mapping[server][thread] gives, for each server, a unique number that addresses the device
	 *
	 */
	struct gunrock::app::GPU_Topology read_mpi_topology_config_file(const char * filename)
	{
	    std::vector< std::vector< pair<int,int> > > gpu_mapping(0);
	    int global_gpu_nr=0;
	    ifstream conf_file(filename);
	    if(conf_file.is_open())
	    {
	        std::string line;
	        while(conf_file.good())
	        {
	            getline(conf_file, line);
	            std::vector<std::string> token = tokenize(line);
	            if(token.size() > 1 && token[0][0]!='#')
	            {
	                std::string name = token[0];
	                int target_server  = atoi(token[1].c_str());
	                if(gpu_mapping.size() <= target_server)
	                {
	                    gpu_mapping.resize(target_server+1);
	                }
                
	                for(size_t i=0;i<token.size()-2;i++)
	                {
	                    int gpu_nr = atoi(token[i+2].c_str());
	                    gpu_mapping[target_server].push_back(make_pair(gpu_nr,global_gpu_nr));
	                    global_gpu_nr++;
	                }
	            }
	        }
	    }
	    else
	    {
	        fprintf(stderr,"unable to open mpi topology configuation file %s\n",filename);
	        exit(-1);
	    }
    
	    conf_file.close();
    
	    struct gunrock::app::GPU_Topology topol;
	    topol.num_servers = (int)gpu_mapping.size();
	    topol.num_gpus_per_server = (int  *)malloc(sizeof(int)*topol.num_servers);
	    topol.rank_of_gpu         = (int  *)malloc(sizeof(int)*global_gpu_nr);
	    topol.local_gpu_mapping   = (int **)malloc(sizeof(int*)*topol.num_servers);
	    topol.global_gpu_maping   = (int **)malloc(sizeof(int*)*topol.num_servers);
	    topol.total_num_gpus      = global_gpu_nr;
	    int offset=0;
	    for(int i=0;i<topol.num_servers; i++)
	    {
	        topol.num_gpus_per_server[i] = (int)gpu_mapping[i].size();
	        topol.local_gpu_mapping[i]   = (int*)malloc(sizeof(int) * topol.num_gpus_per_server[i]);
	        topol.global_gpu_maping[i]   = (int*)malloc(sizeof(int) * topol.num_gpus_per_server[i]);
        
	        for(int j=0; j<topol.num_gpus_per_server[i]; j++)
	        {
	            topol.rank_of_gpu[offset]=i;
	            offset++;
	            topol.local_gpu_mapping[i][j] = (int)gpu_mapping[i][j].first;
	            topol.global_gpu_maping[i][j] = (int)gpu_mapping[i][j].second;
	        }
	    }
	    return topol;
	};


	template<typename Type>
	struct type_mpi_struct{
	    Type v;
	};

	template<typename Type>
	void register_mpi_datatypes(MPI_Datatype * t){
	    int lengths[1]={sizeof(Type)};
	    MPI_Aint offsets[1];
	    offsets[0]= offsetof(type_mpi_struct<Type>,v);
	    MPI_Datatype types[1] = {MPI_BYTE};
    
	    MPI_Type_create_struct(1, lengths, offsets, types, t);
	    MPI_Type_commit(t);
		
	}

	#endif


	/*
	 * @brief Base class for graph problem enactors.
 
	 */
	template <
	typename SizeT,
	bool     _DEBUG,  // if DEBUG is set, print details to stdout
	bool     _SIZE_CHECK>
	class EnactorBase
	{
	public:
	    static const bool DEBUG = _DEBUG;
	    static const bool SIZE_CHECK = _SIZE_CHECK;
	    int           num_gpus;
	    int          *gpu_idx;
	    int           num_gpus_local;
	    int           num_gpus_global;
	    int           *gpu_idx_local;
    
	    FrontierType  frontier_type;
    
	    //Device properties
	    util::Array1D<SizeT, util::CudaProperties>          cuda_props        ;
    
	    // Queue size counters and accompanying functionality
	    util::Array1D<SizeT, util::CtaWorkProgressLifetime> work_progress     ;
	    util::Array1D<SizeT, EnactorStats>                  enactor_stats     ;
	    util::Array1D<SizeT, FrontierAttribute<SizeT> >     frontier_attribute;
    
	    FrontierType GetFrontierType() {return frontier_type;}
    
	protected:
    
	    /*
	     * @brief Constructor
	     *
	     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
	     * @param[in] DEBUG If set, will collect kernel running stats and display the running info.
	     */
	    EnactorBase(
	                FrontierType  frontier_type,
	                int           num_gpus,
	                int          *gpu_idx)
	    {
	        this->frontier_type = frontier_type;
	        this->num_gpus      = num_gpus;
	        this->num_gpus_local = num_gpus;
	        this->num_gpus_global = num_gpus;
	        this->gpu_idx       = gpu_idx;
	        cuda_props        .SetName("cuda_props"        );
	        work_progress     .SetName("work_progress"     );
	        enactor_stats     .SetName("enactor_stats"     );
	        frontier_attribute.SetName("frontier_attribute");
	        cuda_props        .Init(num_gpus         , util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
	        work_progress     .Init(num_gpus*num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
	        enactor_stats     .Init(num_gpus*num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
	        frontier_attribute.Init(num_gpus*num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
        
	        for (int gpu=0;gpu<num_gpus;gpu++)
	        {
	            if (util::SetDevice(gpu_idx[gpu])) return;
	            // Setup work progress (only needs doing once since we maintain
	            // it in our kernel code)
	            cuda_props   [gpu].Setup(gpu_idx[gpu]);
	            for (int peer=0;peer<num_gpus;peer++)
	            {
	                work_progress     [gpu*num_gpus+peer].Setup();
	                frontier_attribute[gpu*num_gpus+peer].output_length.Allocate(1, util::HOST | util::DEVICE);
	            }
	        }
	    }
    
	#ifdef WITHMPI
	    /*
	     * @brief Constructor
	     *
	     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
	     * @param[in] DEBUG If set, will collect kernel running stats and display the running info.
	     */
	    EnactorBase(
	                FrontierType  frontier_type,
	                int           num_gpus_global,
	                int          *gpu_idx_local,
	                int 			 num_gpu_local
	                )
	    {
	        /* num_gpu is ambiguous for mulit-node-multi-gpu since it can either mean total number or local node gpus number.
	         * due to this ambiguity it should not be used.
	         */
	        this->gpu_idx=NULL;
        
	        this->frontier_type        = frontier_type;
	        this->num_gpus_global      = num_gpus_global;
	        this->gpu_idx_local        = gpu_idx_local;
	        cuda_props        .SetName("cuda_props"        );
	        work_progress     .SetName("work_progress"     );
	        enactor_stats     .SetName("enactor_stats"     );
	        frontier_attribute.SetName("frontier_attribute");
	        cuda_props        .Init(num_gpu_local                , util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
	        work_progress     .Init(num_gpu_local*num_gpus_global, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
	        enactor_stats     .Init(num_gpu_local*num_gpus_global, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
	        frontier_attribute.Init(num_gpu_local*num_gpus_global, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
        
	        for (int gpu=0; gpu<num_gpu_local; gpu++)
	        {
	            if (util::SetDevice(gpu_idx_local[gpu])) return;
	            // Setup work progress (only needs doing once since we maintain
	            // it in our kernel code)
	            cuda_props   [gpu].Setup(gpu_idx_local[gpu]);
	            for (int peer=0; peer<num_gpus_global; peer++)
	            {
	                work_progress     [gpu*num_gpus_global + peer].Setup();
	                frontier_attribute[gpu*num_gpus_global + peer].output_length.Allocate(1, util::HOST | util::DEVICE);
	            }
	        }
	    }
    
	#endif
    
    
    
	    /*
	     * @brief Destructor
	     */
	    virtual ~EnactorBase()
	    {
	        for (int gpu=0;gpu<num_gpus;gpu++)
	        {
	            if (util::SetDevice(gpu_idx[gpu])) return;
	            for (int peer=0;peer<num_gpus;peer++)
	            {
	                enactor_stats     [gpu*num_gpus+peer].node_locks    .Release();
	                enactor_stats     [gpu*num_gpus+peer].node_locks_out.Release();
	                enactor_stats     [gpu*num_gpus+peer].total_queued  .Release();
	                frontier_attribute[gpu*num_gpus+peer].output_length .Release();
	                if (work_progress [gpu*num_gpus+peer].HostReset()) return;
	            }
	        }
	        work_progress     .Release();
	        cuda_props        .Release();
	        enactor_stats     .Release();
	        frontier_attribute.Release();
	    }
    
	    /*
	     * @brief Init function for enactor base class
	     *
	     * @tparam ProblemData
	     *
	     * @param[in] problem The problem object for the graph primitive
	     * @param[in] max_grid_size Maximum CUDA block numbers in on grid
	     * @param[in] advance_occupancy CTA Occupancy for Advance operator
	     * @param[in] filter_occupancy CTA Occupancy for Filter operator
	     * @param[in] node_lock_size The size of an auxiliary array used in enactor, 256 by default.
	     * \return cudaError_t object which indicates the success of all CUDA function calls.
	     */
	    template <typename Problem>
	    cudaError_t Init(
	                     Problem *problem,
	                     int max_grid_size,
	                     int advance_occupancy,
	                     int filter_occupancy,
	                     int node_lock_size = 256)
	    {
	        cudaError_t retval = cudaSuccess;
        
        
	        for (int gpu=0; gpu<num_gpus_local ;gpu++)
	        {
	            if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
	            for (int peer=0; peer<num_gpus_global ;peer++)
	            {
	                EnactorStats *enactor_stats_ = enactor_stats + gpu*num_gpus_global + peer;
	                //initialize runtime stats
	                enactor_stats_ -> advance_grid_size = MaxGridSize(gpu, advance_occupancy, max_grid_size);
	                enactor_stats_ -> filter_grid_size  = MaxGridSize(gpu, filter_occupancy, max_grid_size);
                
	                if (retval = enactor_stats_ -> advance_kernel_stats.Setup(enactor_stats_->advance_grid_size)) return retval;
	                if (retval = enactor_stats_ ->  filter_kernel_stats.Setup(enactor_stats_->filter_grid_size)) return retval;
	                if (retval = enactor_stats_ -> node_locks    .Allocate(node_lock_size, util::DEVICE)) return retval;
	                if (retval = enactor_stats_ -> node_locks_out.Allocate(node_lock_size, util::DEVICE)) return retval;
	                if (retval = enactor_stats_ -> total_queued  .Allocate(1, util::DEVICE | util::HOST)) return retval;
	            }
	        }
	        return retval;
	    }
    
	    cudaError_t Reset()
	    {
	        cudaError_t retval = cudaSuccess;
        
	        for (int gpu=0; gpu<num_gpus_local; gpu++)
	        {
	            if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
	            for (int peer=0; peer<num_gpus_global; peer++)
	            {
	                EnactorStats *enactor_stats_ = enactor_stats + gpu*num_gpus_global + peer;
	                enactor_stats_ -> iteration             = 0;
	                enactor_stats_ -> total_runtimes        = 0;
	                enactor_stats_ -> total_lifetimes       = 0;
	                enactor_stats_ -> total_queued[0]       = 0;
	                enactor_stats_ -> total_queued.Move(util::HOST, util::DEVICE);
	            }
	        }
	        return retval;
	    }
    
	    template <typename Problem>
	    cudaError_t Setup(
	                      Problem *problem,
	                      int max_grid_size,
	                      int advance_occupancy,
	                      int filter_occupancy,
	                      int node_lock_size = 256)
	    {
	        cudaError_t retval = cudaSuccess;
        
	        if (retval = Init(problem, max_grid_size, advance_occupancy, filter_occupancy, node_lock_size)) return retval;
	        if (retval = Reset()) return retval;
	        return retval;
	    }
    
	    /*
	     * @brief Utility function for getting the max grid size.
	     *
	     * @param[in] cta_occupancy CTA occupancy for current architecture
	     * @param[in] max_grid_size Preset max grid size. If less or equal to 0, fully populate all SMs
	     *
	     * \return The maximum number of threadblocks this enactor class can launch.
	     */
	    int MaxGridSize(int gpu, int cta_occupancy, int max_grid_size = 0)
	    {
	        if (max_grid_size <= 0) {
	            max_grid_size = this->cuda_props[gpu].device_props.multiProcessorCount * cta_occupancy;
	        }
        
	        return max_grid_size;
	    }
	};

	template <
	typename AdvanceKernelPolicy,
	typename FilterKernelPolicy,
	typename Enactor,
	bool     _HAS_SUBQ,
	bool     _HAS_FULLQ,
	bool     _BACKWARD,
	bool     _FORWARD,
	bool     _UPDATE_PREDECESSORS>
	struct IterationBase
	{
	public:
	    typedef typename Enactor::SizeT      SizeT     ;
	    typedef typename Enactor::Value      Value     ;
	    typedef typename Enactor::VertexId   VertexId  ;
	    typedef typename Enactor::Problem    Problem   ;
	    typedef typename Problem::DataSlice  DataSlice ;
	    typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;
	    static const bool INSTRUMENT = Enactor::INSTRUMENT;
	    static const bool DEBUG      = Enactor::DEBUG;
	    static const bool SIZE_CHECK = Enactor::SIZE_CHECK;
	    static const bool HAS_SUBQ   = _HAS_SUBQ;
	    static const bool HAS_FULLQ  = _HAS_FULLQ;
	    static const bool BACKWARD   = _BACKWARD;
	    static const bool FORWARD    = _FORWARD;
	    static const bool UPDATE_PREDECESSORS = _UPDATE_PREDECESSORS;
    
	    static void SubQueue_Gather(
	                                int                            thread_num,
	                                int                            peer_,
	                                util::DoubleBuffer<SizeT, VertexId, Value>
	                                *frontier_queue,
	                                util::Array1D<SizeT, SizeT>   *scanned_edges,
	                                FrontierAttribute<SizeT>      *frontier_attribute,
	                                EnactorStats                  *enactor_stats,
	                                DataSlice                     *data_slice,
	                                DataSlice                     *d_data_slice,
	                                GraphSlice                    *graph_slice,
	                                util::CtaWorkProgressLifetime *work_progress,
	                                ContextPtr                     context,
	                                cudaStream_t                   stream)
	    {
	    }
    
	    static void SubQueue_Core(
	                              int                            thread_num,
	                              int                            peer_,
	                              util::DoubleBuffer<SizeT, VertexId, Value>
	                              *frontier_queue,
	                              util::Array1D<SizeT, SizeT>   *scanned_edges,
	                              FrontierAttribute<SizeT>      *frontier_attribute,
	                              EnactorStats                  *enactor_stats,
	                              DataSlice                     *data_slice,
	                              DataSlice                     *d_data_slice,
	                              GraphSlice                    *graph_slice,
	                              util::CtaWorkProgressLifetime *work_progress,
	                              ContextPtr                     context,
	                              cudaStream_t                   stream)
	    {
	    }
    
	    static void FullQueue_Gather(
	                                 int                            thread_num,
	                                 int                            peer_,
	                                 util::DoubleBuffer<SizeT, VertexId, Value>
	                                 *frontier_queue,
	                                 util::Array1D<SizeT, SizeT>   *scanned_edges,
	                                 FrontierAttribute<SizeT>      *frontier_attribute,
	                                 EnactorStats                  *enactor_stats,
	                                 DataSlice                     *data_slice,
	                                 DataSlice                     *d_data_slice,
	                                 GraphSlice                    *graph_slice,
	                                 util::CtaWorkProgressLifetime *work_progress,
	                                 ContextPtr                     context,
	                                 cudaStream_t                   stream)
	    {
	    }
    
	    static void FullQueue_Core(
	                               int                            thread_num,
	                               int                            peer_,
	                               util::DoubleBuffer<SizeT, VertexId, Value>
	                               *frontier_queue,
	                               util::Array1D<SizeT, SizeT>   *scanned_edges,
	                               FrontierAttribute<SizeT>      *frontier_attribute,
	                               EnactorStats                  *enactor_stats,
	                               DataSlice                     *data_slice,
	                               DataSlice                     *d_data_slice,
	                               GraphSlice                    *graph_slice,
	                               util::CtaWorkProgressLifetime *work_progress,
	                               ContextPtr                     context,
	                               cudaStream_t                   stream)
	    {
	    }
    
	    static bool Stop_Condition(
	                               EnactorStats                  *enactor_stats,
	                               FrontierAttribute<SizeT>      *frontier_attribute,
	                               util::Array1D<SizeT, DataSlice>
	                               *data_slice,
	                               int                            num_gpus_local,
	                               int                            num_gpus_global
	                               )
	    {
	        return All_Done(enactor_stats, frontier_attribute, data_slice, num_gpus_local, num_gpus_global);
	    }
    
	    static void Iteration_Change(long long &iterations)
	    {
	        iterations++;
	    }
    
	    static void Iteration_Update_Preds(
	                                       GraphSlice                    *graph_slice,
	                                       DataSlice                     *data_slice,
	                                       FrontierAttribute<SizeT>
	                                       *frontier_attribute,
	                                       util::DoubleBuffer<SizeT, VertexId, Value>
	                                       *frontier_queue,
	                                       SizeT                          num_elements,
	                                       cudaStream_t                   stream)
	    {
	        if (num_elements == 0) return;
	        int selector    = frontier_attribute->selector;
	        int grid_size   = num_elements / 256;
	        if ((num_elements % 256) !=0) grid_size++;
	        if (grid_size > 512) grid_size = 512;
        
	        if (Problem::MARK_PREDECESSORS && UPDATE_PREDECESSORS && num_elements>0 )
	        {
	            Copy_Preds<VertexId, SizeT> <<<grid_size,256,0, stream>>>(
	                                                                      num_elements,
	                                                                      frontier_queue->keys[selector].GetPointer(util::DEVICE),
	                                                                      data_slice    ->preds         .GetPointer(util::DEVICE),
	                                                                      data_slice    ->temp_preds    .GetPointer(util::DEVICE));
            
	            Update_Preds<VertexId,SizeT> <<<grid_size,256,0,stream>>>(
	                                                                      num_elements,
	                                                                      graph_slice   ->nodes,
	                                                                      frontier_queue->keys[selector] .GetPointer(util::DEVICE),
	                                                                      graph_slice   ->original_vertex.GetPointer(util::DEVICE),
	                                                                      data_slice    ->temp_preds     .GetPointer(util::DEVICE),
	                                                                      data_slice    ->preds          .GetPointer(util::DEVICE));//,
	        }
	    }
    
	    static void Check_Queue_Size(
	                                 int                            thread_num,
	                                 int                            peer_,
	                                 SizeT                          request_length,
	                                 util::DoubleBuffer<SizeT, VertexId, Value>
	                                 *frontier_queue,
	                                 FrontierAttribute<SizeT>      *frontier_attribute,
	                                 EnactorStats                  *enactor_stats,
	                                 GraphSlice                    *graph_slice)
	    {
	        bool over_sized = false;
	        int  selector   = frontier_attribute->selector;
	        int  iteration  = enactor_stats -> iteration;
        
	        if (Enactor::DEBUG)
	            printf("%d\t %d\t %d\t queue_length = %d, output_length = %d\n",
	                   thread_num, iteration, peer_,
	                   frontier_queue->keys[selector^1].GetSize(),
	                   request_length);fflush(stdout);
        
	        if (enactor_stats->retval =
	            Check_Size<true, SizeT, VertexId > ("queue3", request_length, &frontier_queue->keys  [selector^1], over_sized, thread_num, iteration, peer_, false)) return;
	        if (enactor_stats->retval =
	            Check_Size<true, SizeT, VertexId > ("queue3", request_length, &frontier_queue->keys  [selector  ], over_sized, thread_num, iteration, peer_, true )) return;
	        if (Problem::USE_DOUBLE_BUFFER)
	        {
	            if (enactor_stats->retval =
	                Check_Size<true, SizeT, Value> ("queue3", request_length, &frontier_queue->values[selector^1], over_sized, thread_num, iteration, peer_, false)) return;
	            if (enactor_stats->retval =
	                Check_Size<true, SizeT, Value> ("queue3", request_length, &frontier_queue->values[selector  ], over_sized, thread_num, iteration, peer_, true )) return;
	        }
	    }
    
	    template <
	    int NUM_VERTEX_ASSOCIATES,
	    int NUM_VALUE__ASSOCIATES>
	    static void Make_Output(
	                            int                            thread_num,
	                            SizeT                          num_elements,
	                            int                            num_gpus,
	                            util::DoubleBuffer<SizeT, VertexId, Value>
	                            *frontier_queue,
	                            util::Array1D<SizeT, SizeT>   *scanned_edges,
	                            FrontierAttribute<SizeT>      *frontier_attribute,
	                            EnactorStats                  *enactor_stats,
	                            util::Array1D<SizeT, DataSlice>
	                            *data_slice_,
	                            GraphSlice                    *graph_slice,
	                            util::CtaWorkProgressLifetime *work_progress,
	                            ContextPtr                     context,
	                            cudaStream_t                   stream)
	    {
	        if (num_gpus < 2) return;
	        bool over_sized = false, keys_over_sized = false;
	        int peer_ = 0, t=0, i=0;
	        size_t offset = 0;
	        SizeT *t_out_length = new SizeT[num_gpus];
	        int selector = frontier_attribute->selector;
	        int block_size = 256;
	        int grid_size  = num_elements / block_size;
	        if ((num_elements % block_size)!=0) grid_size ++;
	        if (grid_size > 512) grid_size=512;
	        DataSlice* data_slice=data_slice_->GetPointer(util::HOST);
        
	        for (peer_ = 0; peer_<num_gpus; peer_++)
	        {
	            t_out_length[peer_] = 0;
	            data_slice->out_length[peer_] = 0;
	        }
	        if (num_elements ==0) return;
        
	        over_sized = false;
	        for (peer_ = 0; peer_<num_gpus; peer_++)
	        {
	            if (enactor_stats->retval =
	                Check_Size<Enactor::SIZE_CHECK, SizeT, SizeT> ("keys_marker", num_elements, &data_slice->keys_marker[peer_], over_sized, thread_num, enactor_stats->iteration, peer_)) break;
	            if (over_sized) data_slice->keys_markers[peer_]=data_slice->keys_marker[peer_].GetPointer(util::DEVICE);
	        }
	        if (enactor_stats->retval) return;
	        if (over_sized) data_slice->keys_markers.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);
        
	        for (t=0; t<2; t++)
	        {
	            if (t==0 && !FORWARD) continue;
	            if (t==1 && !BACKWARD) continue;
            
	            if (BACKWARD && t==1)
	                Assign_Marker_Backward<VertexId, SizeT>
	                <<<grid_size, block_size, num_gpus * sizeof(SizeT*) ,stream>>> (
	                                                                                num_elements,
	                                                                                num_gpus,
	                                                                                frontier_queue->keys[selector]    .GetPointer(util::DEVICE),
	                                                                                graph_slice   ->backward_offset   .GetPointer(util::DEVICE),
	                                                                                graph_slice   ->backward_partition.GetPointer(util::DEVICE),
	                                                                                data_slice    ->keys_markers      .GetPointer(util::DEVICE));
	            else if (FORWARD && t==0)
	                Assign_Marker<VertexId, SizeT>
	                <<<grid_size, block_size, num_gpus * sizeof(SizeT*) ,stream>>> (
	                                                                                num_elements,
	                                                                                num_gpus,
	                                                                                frontier_queue->keys[selector]    .GetPointer(util::DEVICE),
	                                                                                graph_slice   ->partition_table   .GetPointer(util::DEVICE),
	                                                                                data_slice    ->keys_markers      .GetPointer(util::DEVICE));
            
	            for (peer_=0;peer_<num_gpus;peer_++)
	            {
	                Scan<mgpu::MgpuScanTypeInc>(
	                                            (SizeT*)data_slice->keys_marker[peer_].GetPointer(util::DEVICE),
	                                            num_elements,
	                                            (SizeT)0, mgpu::plus<SizeT>(), (SizeT*)0, (SizeT*)0,
	                                            (SizeT*)data_slice->keys_marker[peer_].GetPointer(util::DEVICE),
	                                            context[0]);
	            }
            
	            if (num_elements>0) for (peer_=0; peer_<num_gpus;peer_++)
	            {
	                cudaMemcpyAsync(&(t_out_length[peer_]),
	                                data_slice->keys_marker[peer_].GetPointer(util::DEVICE)
	                                + (num_elements -1),
	                                sizeof(SizeT), cudaMemcpyDeviceToHost, stream);
	            } else {
	                for (peer_=0;peer_<num_gpus;peer_++)
	                    t_out_length[peer_]=0;
	            }
	            if (enactor_stats->retval = cudaStreamSynchronize(stream)) break;
            
	            keys_over_sized = true;
	            for (peer_=0; peer_<num_gpus;peer_++)
	            {
	                if (enactor_stats->retval =
	                    Check_Size <Enactor::SIZE_CHECK, SizeT, VertexId> (
	                                                                       "keys_out",
	                                                                       data_slice->out_length[peer_] + t_out_length[peer_],
	                                                                       peer_!=0 ? &data_slice->keys_out[peer_] :
	                                                                       &data_slice->frontier_queues[0].keys[selector^1],
	                                                                       keys_over_sized, thread_num, enactor_stats[0].iteration, peer_),
	                    data_slice->out_length[peer_]==0? false: true) break;
	                if (keys_over_sized)
	                    data_slice->keys_outs[peer_] = peer_==0 ?
	                    data_slice->frontier_queues[0].keys[selector^1].GetPointer(util::DEVICE) :
	                    data_slice->keys_out[peer_].GetPointer(util::DEVICE);
	                if (peer_ == 0) continue;
                
	                over_sized = false;
	                for (i=0;i<NUM_VERTEX_ASSOCIATES;i++)
	                {
	                    if (enactor_stats[0].retval =
	                        Check_Size <Enactor::SIZE_CHECK, SizeT, VertexId>(
	                                                                          "vertex_associate_outs",
	                                                                          data_slice->out_length[peer_] + t_out_length[peer_],
	                                                                          &data_slice->vertex_associate_out[peer_][i],
	                                                                          over_sized, thread_num, enactor_stats->iteration, peer_),
	                        data_slice->out_length[peer_]==0? false: true) break;
	                    if (over_sized) data_slice->vertex_associate_outs[peer_][i] = data_slice->vertex_associate_out[peer_][i].GetPointer(util::DEVICE);
	                }
	                if (enactor_stats->retval) break;
	                if (over_sized) data_slice->vertex_associate_outs[peer_].Move(util::HOST, util::DEVICE, NUM_VERTEX_ASSOCIATES, 0, stream);
                
	                over_sized = false;
	                for (i=0;i<NUM_VALUE__ASSOCIATES;i++)
	                {
	                    if (enactor_stats->retval =
	                        Check_Size<Enactor::SIZE_CHECK, SizeT, Value   >(
	                                                                         "value__associate_outs",
	                                                                         data_slice->out_length[peer_] + t_out_length[peer_],
	                                                                         &data_slice->value__associate_out[peer_][i],
	                                                                         over_sized, thread_num, enactor_stats->iteration, peer_,
	                                                                         data_slice->out_length[peer_]==0? false: true)) break;
	                    if (over_sized) data_slice->value__associate_outs[peer_][i] = data_slice->value__associate_out[peer_][i].GetPointer(util::DEVICE);
	                }
	                if (enactor_stats->retval) break;
	                if (over_sized) data_slice->value__associate_outs[peer_].Move(util::HOST, util::DEVICE, NUM_VALUE__ASSOCIATES, 0, stream);
	            }
	            if (enactor_stats->retval) break;
	            if (keys_over_sized) data_slice->keys_outs.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);
            
	            offset = 0;
	            memcpy(&(data_slice -> make_out_array[offset]),
	                   data_slice -> keys_markers         .GetPointer(util::HOST),
	                   sizeof(SizeT*   ) * num_gpus);
	            offset += sizeof(SizeT*   ) * num_gpus ;
	            memcpy(&(data_slice -> make_out_array[offset]),
	                   data_slice -> keys_outs            .GetPointer(util::HOST),
	                   sizeof(VertexId*) * num_gpus);
	            offset += sizeof(VertexId*) * num_gpus ;
	            memcpy(&(data_slice -> make_out_array[offset]),
	                   data_slice -> vertex_associate_orgs.GetPointer(util::HOST),
	                   sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
	            offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
	            memcpy(&(data_slice -> make_out_array[offset]),
	                   data_slice -> value__associate_orgs.GetPointer(util::HOST),
	                   sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
	            offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;
	            for (peer_=0; peer_<num_gpus; peer_++)
	            {
	                memcpy(&(data_slice->make_out_array[offset]),
	                       data_slice->vertex_associate_outs[peer_].GetPointer(util::HOST),
	                       sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
	                offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
	            }
	            for (peer_=0; peer_<num_gpus; peer_++)
	            {
	                memcpy(&(data_slice->make_out_array[offset]),
	                       data_slice->value__associate_outs[peer_].GetPointer(util::HOST),
	                       sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
	                offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;
	            }
	            memcpy(&(data_slice->make_out_array[offset]),
	                   data_slice->out_length.GetPointer(util::HOST),
	                   sizeof(SizeT) * num_gpus);
	            offset += sizeof(SizeT) * num_gpus;
	            data_slice->make_out_array.Move(util::HOST, util::DEVICE, offset, 0, stream);
            
	            if (BACKWARD && t==1)
	                Make_Out_Backward<VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
	                <<<grid_size, block_size, sizeof(char)*offset, stream>>> (
	                                                                          num_elements,
	                                                                          num_gpus,
	                                                                          frontier_queue-> keys[selector]      .GetPointer(util::DEVICE),
	                                                                          graph_slice   -> backward_offset     .GetPointer(util::DEVICE),
	                                                                          graph_slice   -> backward_partition  .GetPointer(util::DEVICE),
	                                                                          graph_slice   -> backward_convertion .GetPointer(util::DEVICE),
	                                                                          offset,
	                                                                          data_slice    -> make_out_array      .GetPointer(util::DEVICE));
	            else if (FORWARD && t==0)
	                Make_Out<VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
	                <<<grid_size, block_size, sizeof(char)*offset, stream>>> (
	                                                                          num_elements,
	                                                                          num_gpus,
	                                                                          frontier_queue-> keys[selector]      .GetPointer(util::DEVICE),
	                                                                          graph_slice   -> partition_table     .GetPointer(util::DEVICE),
	                                                                          graph_slice   -> convertion_table    .GetPointer(util::DEVICE),
	                                                                          offset,
	                                                                          data_slice    -> make_out_array      .GetPointer(util::DEVICE));
	            for (peer_ = 0; peer_<num_gpus; peer_++)
	                data_slice->out_length[peer_] += t_out_length[peer_];
	        }
	        if (enactor_stats->retval) return;
	        if (enactor_stats->retval = cudaStreamSynchronize(stream)) return;
	        frontier_attribute->selector^=1;
	        if (t_out_length!=NULL) {delete[] t_out_length; t_out_length=NULL;}
	    }
    
	};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
