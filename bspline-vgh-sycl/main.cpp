#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "common.h"

#define max(a,b) ((a<b)?b:a)
#define min(a,b) ((a<b)?a:b)

const int WSIZE = 12000;          // Walker
const int NSIZE = 2003;           // Values
const int MSIZE = NSIZE*3+3;      // Gradient vectors
const int OSIZE = NSIZE*9+9;      // Hessian Matrices 

const int NSIZE_round = NSIZE%16 ? NSIZE+16-NSIZE%16: NSIZE;
const size_t SSIZE = (size_t)NSIZE_round*48*48*48;  //Coefs size 

void eval_abc(const float *Af, float tx, float *a) {

  a[0] = ( ( Af[0]  * tx + Af[1] ) * tx + Af[2] ) * tx + Af[3];
  a[1] = ( ( Af[4]  * tx + Af[5] ) * tx + Af[6] ) * tx + Af[7];
  a[2] = ( ( Af[8]  * tx + Af[9] ) * tx + Af[10] ) * tx + Af[11];
  a[3] = ( ( Af[12] * tx + Af[13] ) * tx + Af[14] ) * tx + Af[15];
}

static inline void eval_UBspline_3d_s_vgh (
    const float * __restrict coefs_init,
    const intptr_t xs,
    const intptr_t ys,
    const intptr_t zs,
    float * __restrict vals,
    float * __restrict grads,
    float * __restrict hess,
    const float *   a, const float *   b, const float *   c,
    const float *  da, const float *  db, const float *  dc,
    const float * d2a, const float * d2b, const float * d2c,
    const float dxInv, const float dyInv, const float dzInv)
{
  float h[9];
  float v0 = 0.0f;
  for (int i = 0; i < 9; ++i) h[i] = 0.0f;

  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++) {
      float pre20 = d2a[i]*  b[j];
      float pre10 =  da[i]*  b[j];
      float pre00 =   a[i]*  b[j];
      float pre11 =  da[i]* db[j];
      float pre01 =   a[i]* db[j];
      float pre02 =   a[i]*d2b[j];

      const float *coefs = coefs_init + i*xs + j*ys;

      float sum0 =   c[0] * coefs[0] +   c[1] * coefs[zs] +   c[2] * coefs[zs*2] +   c[3] * coefs[zs*3];
      float sum1 =  dc[0] * coefs[0] +  dc[1] * coefs[zs] +  dc[2] * coefs[zs*2] +  dc[3] * coefs[zs*3];
      float sum2 = d2c[0] * coefs[0] + d2c[1] * coefs[zs] + d2c[2] * coefs[zs*2] + d2c[3] * coefs[zs*3];

      h[0]  += pre20 * sum0;
      h[1]  += pre11 * sum0;
      h[2]  += pre10 * sum1;
      h[4]  += pre02 * sum0;
      h[5]  += pre01 * sum1;
      h[8]  += pre00 * sum2;
      h[3]  += pre10 * sum0;
      h[6]  += pre01 * sum0;
      h[7]  += pre00 * sum1;
      v0    += pre00 * sum0;
    }
  vals[0] = v0;
  grads[0]  = h[3] * dxInv;
  grads[1]  = h[6] * dyInv;
  grads[2]  = h[7] * dzInv;

  hess [0] = h[0]*dxInv*dxInv;
  hess [1] = h[1]*dxInv*dyInv;
  hess [2] = h[2]*dxInv*dzInv;
  hess [3] = h[1]*dxInv*dyInv; // Copy hessian elements into lower half of 3x3 matrix
  hess [4] = h[4]*dyInv*dyInv;
  hess [5] = h[5]*dyInv*dzInv;
  hess [6] = h[2]*dxInv*dzInv; // Copy hessian elements into lower half of 3x3 matrix
  hess [7] = h[5]*dyInv*dzInv; //Copy hessian elements into lower half of 3x3 matrix
  hess [8] = h[8]*dzInv*dzInv;
}

int main(int argc, char ** argv) {

  float *Af = (float*) malloc (sizeof(float)*16);
  float *dAf = (float*) malloc (sizeof(float)*16);
  float *d2Af = (float*) malloc (sizeof(float)*16);

  Af[0]=-0.166667;
  Af[1]=0.500000;
  Af[2]=-0.500000;
  Af[3]=0.166667;
  Af[4]=0.500000;
  Af[5]=-1.000000;
  Af[6]=0.000000;
  Af[7]=0.666667;
  Af[8]=-0.500000;
  Af[9]=0.500000;
  Af[10]=0.500000;
  Af[11]=0.166667;
  Af[12]=0.166667;
  Af[13]=0.000000;
  Af[14]=0.000000;
  Af[15]=0.000000;
  dAf[0]=0.000000; d2Af[0]=0.000000;
  dAf[1]=-0.500000; d2Af[1]=0.000000;
  dAf[2]=1.000000; d2Af[2]=-1.000000;
  dAf[3]=-0.500000; d2Af[3]=1.000000;
  dAf[4]=0.000000; d2Af[4]=0.000000;
  dAf[5]=1.500000; d2Af[5]=0.000000;
  dAf[6]=-2.000000; d2Af[6]=3.000000;
  dAf[7]=0.000000; d2Af[7]=-2.000000;
  dAf[8]=0.000000; d2Af[8]=0.000000;
  dAf[9]=-1.500000; d2Af[9]=0.000000;
  dAf[10]=1.000000; d2Af[10]=-3.00000;
  dAf[11]=0.500000; d2Af[11]=1.000000;
  dAf[12]=0.000000; d2Af[12]=0.000000;
  dAf[13]=0.500000; d2Af[13]=0.000000;
  dAf[14]=0.000000; d2Af[14]=1.000000;
  dAf[15]=0.000000; d2Af[15]=0.000000;

  float x=0.822387;  
  float y=0.989919;  
  float z=0.104573;

  float* walkers_vals = (float*) malloc(sizeof(float)*WSIZE*NSIZE);
  float* walkers_grads = (float*) malloc(sizeof(float)*WSIZE*MSIZE);
  float* walkers_hess = (float*) malloc(sizeof(float)*WSIZE*OSIZE);
  float* walkers_x = (float*) malloc(sizeof(float)*WSIZE);
  float* walkers_y = (float*) malloc(sizeof(float)*WSIZE);
  float* walkers_z = (float*) malloc(sizeof(float)*WSIZE);

  for (int i=0; i<WSIZE; i++) {
    walkers_x[i] = x + i*1.0/WSIZE;
    walkers_y[i] = y + i*1.0/WSIZE;
    walkers_z[i] = z + i*1.0/WSIZE;
  }

  float* spline_coefs = (float*) malloc (sizeof(float)*SSIZE);
  for(size_t i=0;i<SSIZE;i++)
    spline_coefs[i]=std::sqrt(0.22+i*1.0)*std::sin(i*1.0);

  int spline_num_splines = NSIZE;
  int spline_x_grid_start = 0; 
  int spline_y_grid_start = 0; 
  int spline_z_grid_start = 0; 
  int spline_x_grid_num = 45; 
  int spline_y_grid_num = 45; 
  int spline_z_grid_num = 45; 
  int spline_x_stride=NSIZE_round*48*48;
  int spline_y_stride=NSIZE_round*48;
  int spline_z_stride=NSIZE_round;
  int spline_x_grid_delta_inv=45;
  int spline_y_grid_delta_inv=45;
  int spline_z_grid_delta_inv=45;

  {
#ifdef USE_GPU 
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1> d_walkers_vals (walkers_vals, WSIZE*NSIZE);
  buffer<float, 1> d_walkers_grads (walkers_grads, WSIZE*MSIZE);
  buffer<float, 1> d_walkers_hess (walkers_hess, WSIZE*OSIZE);
  buffer<float, 1> d_spline_coefs (spline_coefs, SSIZE);
  buffer<float, 1> d_a(4);
  buffer<float, 1> d_b(4);
  buffer<float, 1> d_c(4);
  buffer<float, 1> d_da(4);
  buffer<float, 1> d_db(4);
  buffer<float, 1> d_dc(4);
  buffer<float, 1> d_d2a(4);
  buffer<float, 1> d_d2b(4);
  buffer<float, 1> d_d2c(4);

  double total_time = 0.0;

  for(int i=0; i<WSIZE; i++) {
    float x = walkers_x[i], y = walkers_y[i], z = walkers_z[i];

    float ux = x*spline_x_grid_delta_inv;
    float uy = y*spline_y_grid_delta_inv;
    float uz = z*spline_z_grid_delta_inv;
    float ipartx, iparty, ipartz, tx, ty, tz;
    float a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
    intptr_t xs = spline_x_stride;
    intptr_t ys = spline_y_stride;
    intptr_t zs = spline_z_stride;

    x -= spline_x_grid_start;
    y -= spline_y_grid_start;
    z -= spline_z_grid_start;
    ipartx = (int) ux; tx = ux-ipartx; int ix = min(max(0,(int) ipartx),spline_x_grid_num-1);
    iparty = (int) uy; ty = uy-iparty; int iy = min(max(0,(int) iparty),spline_y_grid_num-1);
    ipartz = (int) uz; tz = uz-ipartz; int iz = min(max(0,(int) ipartz),spline_z_grid_num-1);

    eval_abc(Af,tx,&a[0]);

    q.submit([&] (handler &h) {
      auto d_a_acc = d_a.get_access<sycl_discard_write>(h);
      h.copy(a, d_a_acc);
    });

    eval_abc(Af,ty,&b[0]);
    q.submit([&] (handler &h) {
      auto d_b_acc = d_b.get_access<sycl_discard_write>(h);
      h.copy(b, d_b_acc);
    });

    eval_abc(Af,tz,&c[0]);
    q.submit([&] (handler &h) {
      auto d_c_acc = d_c.get_access<sycl_discard_write>(h);
      h.copy(c, d_c_acc);
    });

    eval_abc(dAf,tx,&da[0]);
    q.submit([&] (handler &h) {
      auto d_da_acc = d_da.get_access<sycl_discard_write>(h);
      h.copy(da, d_da_acc);
    });

    eval_abc(dAf,ty,&db[0]);
    q.submit([&] (handler &h) {
      auto d_db_acc = d_db.get_access<sycl_discard_write>(h);
      h.copy(db, d_db_acc);
    });

    eval_abc(dAf,tz,&dc[0]);
    q.submit([&] (handler &h) {
      auto d_dc_acc = d_dc.get_access<sycl_discard_write>(h);
      h.copy(dc, d_dc_acc);
    });

    eval_abc(d2Af,tx,&d2a[0]);
    q.submit([&] (handler &h) {
      auto d_d2a_acc = d_d2a.get_access<sycl_discard_write>(h);
      h.copy(d2a, d_d2a_acc);
    });

    eval_abc(d2Af,ty,&d2b[0]);
    q.submit([&] (handler &h) {
      auto d_d2b_acc = d_d2b.get_access<sycl_discard_write>(h);
      h.copy(d2b, d_d2b_acc);
    });

    eval_abc(d2Af,tz,&d2c[0]);              
    q.submit([&] (handler &h) {
      auto d_d2c_acc = d_d2c.get_access<sycl_discard_write>(h);
      h.copy(d2c, d_d2c_acc);
    });

    range<1> global_size((spline_num_splines+255)/256*256);
    range<1> local_size(256);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    q.submit([&] (handler &h) {
      auto walkers_vals = d_walkers_vals.get_access<sycl_discard_write>(h);
      auto walkers_grads = d_walkers_grads.get_access<sycl_discard_write>(h);
      auto walkers_hess = d_walkers_hess.get_access<sycl_discard_write>(h);
      auto a = d_a.get_access<sycl_read>(h);
      auto b = d_b.get_access<sycl_read>(h);
      auto c = d_c.get_access<sycl_read>(h);
      auto da = d_da.get_access<sycl_read>(h);
      auto db = d_db.get_access<sycl_read>(h);
      auto dc = d_dc.get_access<sycl_read>(h);
      auto d2a = d_d2a.get_access<sycl_read>(h);
      auto d2b = d_d2b.get_access<sycl_read>(h);
      auto d2c = d_d2c.get_access<sycl_read>(h);
      auto spline_coefs = d_spline_coefs.get_access<sycl_read>(h);

      h.parallel_for<class vgh_spline>(nd_range<1>(global_size, local_size), [=] (nd_item<1> item) {
        const int n = item.get_global_id(0);
        if (n < spline_num_splines)
          eval_UBspline_3d_s_vgh ( 
            spline_coefs.get_pointer()+ix*xs+iy*ys+iz*zs+n,
            xs, ys, zs, 
            walkers_vals.get_pointer()+i*NSIZE+n,
            walkers_grads.get_pointer()+i*MSIZE+n*3,
            walkers_hess.get_pointer()+i*OSIZE+n*9,
            a.get_pointer(),
            b.get_pointer(),
            c.get_pointer(),
            da.get_pointer(),
            db.get_pointer(),
            dc.get_pointer(),
            d2a.get_pointer(),
            d2b.get_pointer(),
            d2c.get_pointer(),
            spline_x_grid_delta_inv,
            spline_y_grid_delta_inv,
            spline_z_grid_delta_inv );
        });
    }).wait();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }
  printf("Total kernel execution time %lf (s)\n", total_time * 1e-9);

  }

  // collect results for the first walker
  float resVal = 0.0;
  float resGrad = 0.0;
  float resHess = 0.0;

  for( int i = 0; i < NSIZE; i++ ) resVal = resVal + walkers_vals[i];
  for( int i = 0; i < MSIZE; i++ ) resGrad = resGrad + walkers_grads[i];
  for( int i = 0; i < OSIZE; i++ ) resHess = resHess + walkers_hess[i];
  printf("walkers[0]->collect([resVal resGrad resHess]) = [%e %e %e]\n",
      resVal,resGrad, resHess);  

  free(Af);
  free(dAf);
  free(d2Af);
  free(walkers_vals);
  free(walkers_grads);
  free(walkers_hess);
  free(walkers_x);
  free(walkers_y);
  free(walkers_z);
  free(spline_coefs);

  return 0;
}




