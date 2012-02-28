/*
=========================================================================
main.cu
=========================================================================
This is the main file, from here are coordinated all the operations, namely,
evolution in time and communications between CPU and GPU.

    Copyright (C) 2012  Edison Montoya, eamonto@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


Up to date: 28 Feb 2012					
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "device_functions.h"

#define MAXDIM 3
#define THREADS_PER_BLOCK 64


///////////////////////// GLOBAL VARIABLES///////////////////////

int Num_par;
int INTEGRATOR;
float t,totalenergy,potencial,kinetic,w,G,EPS;
float Num_archive,Num_simulation; 
char *inputfile,*initfile,*outputfile; //IN-OUT VARIABLES 


/////////////PARTICLE STRUCTURE ////////////////////////
float *pos[MAXDIM];
float *vel[MAXDIM];
float *mass;
float *radius;


#include"lib_io.c"
#include"lib_N_body.cu"


int main(int argc,char *argv[]) {

  int j,k,l;
  int N_Blocks;
  int *count;
  float Ttotal,time;

  float *dev_EPS;
  float *dev_G;
  float *dev_t;

  float *dev_mass;

  float *dev_pos0;
  float *dev_pos1;
  float *dev_pos2;

  float *dev_vel0;
  float *dev_vel1;
  float *dev_vel2;

  float *dev_acele0;
  float *dev_acele1;
  float *dev_acele2;

  int *dev_i;


  if(argc != 4) usage(); //VERIFICATION OF INPUT FILES

  initfile=argv[1];      //ASIGNATION
  inputfile=argv[2];     //OF FILE'S 
  outputfile=argv[3];    //NAMES

  create_remove_dir();   //REMOVE OLD outputfile AND CREATE A NEW ONE
  
  read();                //READ INITIAL CONDITIONS

  N_Blocks = Num_par/THREADS_PER_BLOCK;


  //ALLOCATE THE MEMORY ON CPU
  pos[0]=(float *) malloc((size_t) Num_par*sizeof(float));
  pos[1]=(float *) malloc((size_t) Num_par*sizeof(float));
  pos[2]=(float *) malloc((size_t) Num_par*sizeof(float));
  
  vel[0]=(float *) malloc((size_t) Num_par*sizeof(float));
  vel[1]=(float *) malloc((size_t) Num_par*sizeof(float));
  vel[2]=(float *) malloc((size_t) Num_par*sizeof(float));
  
  radius=(float *) malloc((size_t) Num_par*sizeof(float));
  mass  =(float *) malloc((size_t) Num_par*sizeof(float));

  count = (int *) malloc ( sizeof(int) );


  //ALLOCATE THE MEMORY ON GPU
  cudaMalloc( (void**)&dev_i, sizeof(int) );

  cudaMalloc( (void**)&dev_EPS, sizeof(float) );
  cudaMalloc( (void**)&dev_G,   sizeof(float) );
  cudaMalloc( (void**)&dev_t,   sizeof(float) );

  cudaMalloc( (void**)&dev_mass, Num_par*sizeof(float) );

  cudaMalloc( (void**)&dev_pos0, Num_par*sizeof(float) );
  cudaMalloc( (void**)&dev_pos1, Num_par*sizeof(float) );
  cudaMalloc( (void**)&dev_pos2, Num_par*sizeof(float) );

  cudaMalloc( (void**)&dev_vel0, Num_par*sizeof(float) );
  cudaMalloc( (void**)&dev_vel1, Num_par*sizeof(float) );
  cudaMalloc( (void**)&dev_vel2, Num_par*sizeof(float) );

  cudaMalloc( (void**)&dev_acele0, sizeof(float) );
  cudaMalloc( (void**)&dev_acele1, sizeof(float) );
  cudaMalloc( (void**)&dev_acele2, sizeof(float) );


  //READ INITIAL POSITIONS AND VELOCITIES 
  read_input();         


  //Copy memory from CPU to GPU
  cudaMemcpy( dev_EPS, &EPS , sizeof(float),cudaMemcpyHostToDevice );
  cudaMemcpy( dev_G,   &G ,   sizeof(float),cudaMemcpyHostToDevice );
  cudaMemcpy( dev_t,   &t ,   sizeof(float),cudaMemcpyHostToDevice );


  Ttotal = t*Num_simulation*Num_archive;
  printf("\nTotal time of simulation =%f\n\n",Ttotal);

  time=0.0;


  //Copy Mass to the GPU memory
  cudaMemcpy( dev_mass, mass, Num_par*sizeof(float),cudaMemcpyHostToDevice );

  //Copy Positions to the GPU memory
  cudaMemcpy( dev_pos0, pos[0], Num_par*sizeof(float),cudaMemcpyHostToDevice );
  cudaMemcpy( dev_pos1, pos[1], Num_par*sizeof(float),cudaMemcpyHostToDevice );
  cudaMemcpy( dev_pos2, pos[2], Num_par*sizeof(float),cudaMemcpyHostToDevice );

  //Copy Velocities to the GPU memory
  cudaMemcpy( dev_vel0, vel[0], Num_par*sizeof(float),cudaMemcpyHostToDevice );
  cudaMemcpy( dev_vel1, vel[1], Num_par*sizeof(float),cudaMemcpyHostToDevice );
  cudaMemcpy( dev_vel2, vel[2], Num_par*sizeof(float),cudaMemcpyHostToDevice );
    
  
  ////THIS "for" GENERATE "Num_archive" ARCHIVES WITH POSITION AND VELOCITY
  for(k=0 ; k<Num_archive ; k++)
    {
      for (l=0; l<Num_simulation; l++) ///NUMBER OF SIMULATION IN ONE ARCHIVE
	{ 

	  printf("Time of Simulation: %f\n",time);
	  
	  /**************POSITION*****************/
	  cuda_pos<<< N_Blocks, THREADS_PER_BLOCK >>>
	    (dev_t, dev_vel0, dev_vel1, dev_vel2, dev_pos0, dev_pos1, dev_pos2 );
	  
	  
	  cudaThreadSynchronize();	  	  

	  /**************ACELERATION*****************/
	  for(j=0;j<Num_par;j++){
	    
	    count = &j; 
	    
	    cudaMemcpy( dev_i, count, sizeof(int),cudaMemcpyHostToDevice );
	    
	    //Launch cuda_aceleration() kernel with blocks and threads
	    cuda_aceleration<<<  N_Blocks, THREADS_PER_BLOCK >>>
	      ( dev_i, dev_EPS, dev_G, dev_t, dev_mass,
		dev_pos0, dev_pos1, dev_pos2, 
		dev_vel0, dev_vel1, dev_vel2, 
		dev_acele0, dev_acele1, dev_acele2 
		);
	  }

	  cudaThreadSynchronize();

	  /**************POSITION 2*****************/
	  cuda_pos<<< N_Blocks, THREADS_PER_BLOCK >>>
	    (dev_t, dev_vel0, dev_vel1, dev_vel2, dev_pos0, dev_pos1, dev_pos2 );

	  time=time + t;	
	}	  

      //Copy memory from GPU to CPU
      cudaMemcpy( pos[0], dev_pos0, Num_par*sizeof(float), cudaMemcpyDeviceToHost );
      cudaMemcpy( pos[1], dev_pos1, Num_par*sizeof(float), cudaMemcpyDeviceToHost );
      cudaMemcpy( pos[2], dev_pos2, Num_par*sizeof(float), cudaMemcpyDeviceToHost );
      
      write_output(k);
    }
  

  free_CPU_memory();     //FREE CPU MEMORY

  cudaFree(dev_i);       //FREE GPU MEMORY
  
  cudaFree(dev_EPS);     //FREE GPU MEMORY
  cudaFree(dev_G);       //FREE GPU MEMORY
  cudaFree(dev_t);       //FREE GPU MEMORY
  
  cudaFree(dev_mass);    //FREE GPU MEMORY

  cudaFree(dev_pos0);    //FREE GPU MEMORY
  cudaFree(dev_pos1);    //FREE GPU MEMORY
  cudaFree(dev_pos2);    //FREE GPU MEMORY

  cudaFree(dev_vel0);    //FREE GPU MEMORY
  cudaFree(dev_vel1);    //FREE GPU MEMORY
  cudaFree(dev_vel2);    //FREE GPU MEMORY
  
  cudaFree(dev_acele0);  //FREE GPU MEMORY
  cudaFree(dev_acele1);  //FREE GPU MEMORY
  cudaFree(dev_acele2);  //FREE GPU MEMORY
  
  return 0;
}
  

