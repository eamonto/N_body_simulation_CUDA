/*
=========================================================================
lib_io.c
=========================================================================
Into this file are implemented all the input-output routines.

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


///////////VERIFICATION OF INPUT FILES/////////////
int usage(void)
{
  printf("\n");
  printf("USAGE: ./exec <initfile> <inputfile> <outputfile> \n");
  printf("\n");
  exit(1);
}


//REMOVE OLD outputfile AND CREATE A NEW ONE
int create_remove_dir(void)
{
  char remove[200],copy[200],create[200];
  char *rm="rm -rf ",*cp="cp ",*mkdir="mkdir -p ";
  int out_return;

  strcpy(remove,rm);
  strcat(remove,outputfile);
  out_return = system(remove);

  strcpy(create,mkdir);
  strcat(create,outputfile);
  out_return = system(create);

  strcpy(copy,cp);
  strcat(copy,initfile);
  strcat(copy," ");
  strcat(copy,outputfile);
  out_return = system(copy);

  return out_return;
}


////////READ PARAMETERS OF THE SYSTEM/////
int read(void)
{
  int int_out;
  FILE *pf;
  
  pf=fopen(initfile,"r");
  
  int_out = fscanf(pf,"%d %f %f %f %f %f %f %d",&Num_par,&t,&G,&w,
	 &Num_archive,&Num_simulation,&EPS,&INTEGRATOR);
  
  fclose(pf);

  int_out = 0;

  return int_out;
}


//////////////////////READ INITIAL POSITIONS AND VELOCITIES////////////////////
int read_input(void)
{
  int int_out;
  int i;
  FILE *ppf;

  ppf=fopen(inputfile,"r");
  
  for(i=0 ; i<Num_par ; i++)
    {
      int_out = fscanf(ppf,"%f %f %f %f %f %f %f %f",
		       &radius[i],&pos[0][i],&pos[1][i],&pos[2][i],
		       &vel[0][i],&vel[1][i],&vel[2][i],&mass[i]);
    }

  fclose(ppf);

  int_out = 0;

  return int_out;
}


////////////WRITE RADIUS, POSITIONS, VELOCITIES AND MASS/////////////////////
int write_output(int j)
{
  int i;
  char param[30]="0",output[100];  
  FILE *pf;

  gcvt(j,2,param);

  strcpy(output,outputfile);
  strcat(output,"/");
  strcat(output,outputfile);
  strcat(output,"_");
  strcat(output,param);
  strcat(output,".dat");
  
  pf=fopen(output,"w");
  
  for (i=0 ; i < Num_par ; i++)
    {
      radius[i]=sqrt(pos[0][i]*pos[0][i] +
		  pos[1][i]*pos[1][i] +
		  pos[2][i]*pos[2][i]);

      fprintf(pf,"%f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\n",
  	      radius[i],pos[0][i],pos[1][i],pos[2][i],
  	      vel[0][i],vel[1][i],vel[2][i],mass[i]);
    }

  fclose(pf);
  
  return 0;
}


/////////FREE MEMORY THAT WAS ALLOCATE IN THE CPU///////////
int free_CPU_memory(void)
{
  int i;
  
  for(i=0;i<MAXDIM;i++){
    free(pos[i]);
    free(vel[i]);
  }
  
  free(radius);
  free(mass);
    
  return 0;
}

