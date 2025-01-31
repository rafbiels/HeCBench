#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "./util/timer/timer.h"
#include "./util/num/num.h"
#include "common.h"
#include "./main.h"

int main(  int argc, char *argv [])
{
  // counters
  int i, j, k, l, m, n;

  // system memory
  par_str par_cpu;
  dim_str dim_cpu;
  box_str* box_cpu;
  FOUR_VECTOR* rv_cpu;
  fp* qv_cpu;
  FOUR_VECTOR* fv_cpu;
  int nh;

  printf("WG size of kernel = %d \n", NUMBER_THREADS);

  // assing default values
  dim_cpu.arch_arg = 0;
  dim_cpu.cores_arg = 1;
  dim_cpu.boxes1d_arg = 1;

  // go through arguments
  if(argc==3){
    for(dim_cpu.cur_arg=1; dim_cpu.cur_arg<argc; dim_cpu.cur_arg++){
      // check if -boxes1d
      if(strcmp(argv[dim_cpu.cur_arg], "-boxes1d")==0){
        // check if value provided
        if(argc>=dim_cpu.cur_arg+1){
          // check if value is a number
          if(isInteger(argv[dim_cpu.cur_arg+1])==1){
            dim_cpu.boxes1d_arg = atoi(argv[dim_cpu.cur_arg+1]);
            if(dim_cpu.boxes1d_arg<0){
              printf("ERROR: Wrong value to -boxes1d argument, cannot be <=0\n");
              return 0;
            }
            dim_cpu.cur_arg = dim_cpu.cur_arg+1;
          }
          // value is not a number
          else{
            printf("ERROR: Value to -boxes1d argument in not a number\n");
            return 0;
          }
        }
        // value not provided
        else{
          printf("ERROR: Missing value to -boxes1d argument\n");
          return 0;
        }
      }
      // unknown
      else{
        printf("ERROR: Unknown argument\n");
        return 0;
      }
    }
    // Print configuration
    printf("Configuration used: arch = %d, cores = %d, boxes1d = %d\n", dim_cpu.arch_arg, dim_cpu.cores_arg, dim_cpu.boxes1d_arg);
  }
  else{
    printf("Provide boxes1d argument, example: -boxes1d 16");
    return 0;
  }

  par_cpu.alpha = 0.5;

  // total number of boxes
  dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg; 

  // how many particles space has in each direction
  dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
  dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
  dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

  // box array
  dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

  // allocate boxes
  box_cpu = (box_str*)malloc(dim_cpu.box_mem);

  // initialize number of home boxes
  nh = 0;

  // home boxes in z direction
  for(i=0; i<dim_cpu.boxes1d_arg; i++){
    // home boxes in y direction
    for(j=0; j<dim_cpu.boxes1d_arg; j++){
      // home boxes in x direction
      for(k=0; k<dim_cpu.boxes1d_arg; k++){

        // current home box
        box_cpu[nh].x = k;
        box_cpu[nh].y = j;
        box_cpu[nh].z = i;
        box_cpu[nh].number = nh;
        box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

        // initialize number of neighbor boxes
        box_cpu[nh].nn = 0;

        // neighbor boxes in z direction
        for(l=-1; l<2; l++){
          // neighbor boxes in y direction
          for(m=-1; m<2; m++){
            // neighbor boxes in x direction
            for(n=-1; n<2; n++){

              // check if (this neighbor exists) and (it is not the same as home box)
              if(    (((i+l)>=0 && (j+m)>=0 && (k+n)>=0)==true && ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg && (k+n)<dim_cpu.boxes1d_arg)==true)  &&
                  (l==0 && m==0 && n==0)==false  ){

                // current neighbor box
                box_cpu[nh].nei[box_cpu[nh].nn].x = (k+n);
                box_cpu[nh].nei[box_cpu[nh].nn].y = (j+m);
                box_cpu[nh].nei[box_cpu[nh].nn].z = (i+l);
                box_cpu[nh].nei[box_cpu[nh].nn].number =  (box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) + 
                  (box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) + 
                  box_cpu[nh].nei[box_cpu[nh].nn].x;
                box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

                // increment neighbor box
                box_cpu[nh].nn = box_cpu[nh].nn + 1;

              }

            } // neighbor boxes in x direction
          } // neighbor boxes in y direction
        } // neighbor boxes in z direction

        // increment home box
        nh = nh + 1;

      } // home boxes in x direction
    } // home boxes in y direction
  } // home boxes in z direction

  //  PARAMETERS, DISTANCE, CHARGE AND FORCE

  // random generator seed set to random value - time in this case
  srand(2);

  // input (distances)
  rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
  for(i=0; i<dim_cpu.space_elem; i=i+1){
    rv_cpu[i].v = (rand()%10 + 1) / 10.0;      // get a number in the range 0.1 - 1.0
    rv_cpu[i].x = (rand()%10 + 1) / 10.0;      // get a number in the range 0.1 - 1.0
    rv_cpu[i].y = (rand()%10 + 1) / 10.0;      // get a number in the range 0.1 - 1.0
    rv_cpu[i].z = (rand()%10 + 1) / 10.0;      // get a number in the range 0.1 - 1.0
  }

  // input (charge)
  qv_cpu = (fp*)malloc(dim_cpu.space_mem2);
  for(i=0; i<dim_cpu.space_elem; i=i+1){
    qv_cpu[i] = (rand()%10 + 1) / 10.0;      // get a number in the range 0.1 - 1.0
  }

  // output (forces)
  fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
  for(i=0; i<dim_cpu.space_elem; i=i+1){
    fv_cpu[i].v = 0;                // set to 0, because kernels keeps adding to initial value
    fv_cpu[i].x = 0;                // set to 0, because kernels keeps adding to initial value
    fv_cpu[i].y = 0;                // set to 0, because kernels keeps adding to initial value
    fv_cpu[i].z = 0;                // set to 0, because kernels keeps adding to initial value
  }

  long long kstart, kend;
  long long start = get_time();

  { // SYCL scope

#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    //  EXECUTION PARAMETERS
    size_t local_work_size = NUMBER_THREADS;
    size_t global_work_size = dim_cpu.number_boxes * local_work_size;

#ifdef DEBUG
    printf("# of blocks = %lu, # of threads/block = %lu (ensure that device can handle)\n", 
        global_work_size/local_work_size, local_work_size);
#endif

    const property_list props = property::buffer::use_host_ptr();

    int dim_cpu_number_boxes = dim_cpu.number_boxes;

    //  boxes
    buffer<box_str, 1> d_box_gpu(box_cpu, dim_cpu.number_boxes, props);

    //  rv
    buffer<FOUR_VECTOR, 1> d_rv_gpu(rv_cpu, dim_cpu.space_elem, props);

    //  qv
    buffer<fp, 1> d_qv_gpu(qv_cpu, dim_cpu.space_elem, props);

    //  fv
    buffer<FOUR_VECTOR, 1> d_fv_gpu(fv_cpu, dim_cpu.space_elem, props);

    range<1> gws (global_work_size);
    range<1> lws (local_work_size);

    q.wait();
    kstart = get_time();

    q.submit([&](handler& cgh) {
      auto d_box_gpu_acc = d_box_gpu.get_access<sycl_read>(cgh);
      auto d_rv_gpu_acc = d_rv_gpu.get_access<sycl_read>(cgh);
      auto d_qv_gpu_acc = d_qv_gpu.get_access<sycl_read>(cgh);
      auto d_fv_gpu_acc = d_fv_gpu.get_access<sycl_read_write>(cgh);

      accessor <FOUR_VECTOR, 1, sycl_read_write, access::target::local> rA_shared (100, cgh);
      accessor <FOUR_VECTOR, 1, sycl_read_write, access::target::local> rB_shared (100, cgh);
      accessor <fp, 1, sycl_read_write, access::target::local> qB_shared (100, cgh);

      cgh.parallel_for<class lavamd>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        #include "kernel.sycl"
      });
    });

    q.wait();
    kend = get_time();
    
  } // SYCL scope

  long long end = get_time();
  printf("Device offloading time:\n"); 
  printf("%.12f s\n", (float) (end-start) / 1000000); 

  printf("Kernel execution time:\n"); 
  printf("%.12f s\n", (float) (kend-kstart) / 1000000); 

  // dump results
#ifdef OUTPUT
  FILE *fptr;
  fptr = fopen("result.txt", "w");  
  for(i=0; i<dim_cpu.space_elem; i=i+1){
    fprintf(fptr, "%f, %f, %f, %f\n", fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z);
  }
  fclose(fptr);
#endif         


  free(rv_cpu);
  free(qv_cpu);
  free(fv_cpu);
  free(box_cpu);

  return 0; 
}


