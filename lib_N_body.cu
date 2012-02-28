/*
=========================================================================
lib_N_body.cu
=========================================================================
Into this file are implemented the functions that runs on the GPU.

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



//EVOLUTION OF THE POSITIONS USING THE leapfrog ALGORITHM
__global__ void cuda_pos(float *dt, float *vel0, float *vel1, float *vel2, float *pos0, float *pos1, float *pos2)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  pos0[index] = pos0[index] + 0.5*(*dt)*vel0[index];
  pos1[index] = pos1[index] + 0.5*(*dt)*vel1[index];
  pos2[index] = pos2[index] + 0.5*(*dt)*vel2[index];  
}



//////////////// ACELERATION(in variable acele) OF PARTICLE i////////////////
__global__ void cuda_aceleration(int *i, float *eps, float *G, float *dt, float *mass, float *pos0, float *pos1, float *pos2, float *vel0, float *vel1, float *vel2, float *acele0, float *acele1, float *acele2)
{

  // Shared memory for results of multiplication
  __shared__  float Xij[THREADS_PER_BLOCK],Yij[THREADS_PER_BLOCK],Zij[THREADS_PER_BLOCK];

  float X,Y,Z,R;

  int index = threadIdx.x + blockIdx.x * blockDim.x;

  X = pos0[index]-pos0[*i];
  Y = pos1[index]-pos1[*i];
  Z = pos2[index]-pos2[*i];
  
  R = X*X + Y*Y + Z*Z + (*eps)*(*eps);
  
  Xij[threadIdx.x] = X/sqrt(R*R*R);; 
  Yij[threadIdx.x] = Y/sqrt(R*R*R);; 
  Zij[threadIdx.x] = Z/sqrt(R*R*R);; 
  
  __syncthreads(); 
    
  if( 0 ==threadIdx.x ) 
    {
      __shared__ float sumX,sumY,sumZ;

      sumX = 0.0;
      sumY = 0.0;
      sumZ = 0.0;
      
      for( int j = 0; j < THREADS_PER_BLOCK; j++ )
	{
	  sumX += Xij[j];
	  sumY += Yij[j];
	  sumZ += Zij[j];
	  }

      atomicAdd( acele0 , sumX );
      atomicAdd( acele1 , sumY );
      atomicAdd( acele2 , sumZ );

      *acele0 = *acele0*(*G)*mass[*i];
      *acele1 = *acele1*(*G)*mass[*i];
      *acele2 = *acele2*(*G)*mass[*i];      

      if( 0 ==blockIdx.x ) {

	//New velocities
	vel0[*i] = vel0[*i] + (*dt)*(*acele0);
	vel1[*i] = vel1[*i] + (*dt)*(*acele1);
	vel2[*i] = vel2[*i] + (*dt)*(*acele2);
      }
    } 
}

