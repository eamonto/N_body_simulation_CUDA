# PREDEFINED VARIABLES
CFLAGS = -arch sm_20
CC = nvcc

#GLOBAL VARIABLES
INITFILE = init.txt
INPUTFILE = input.dat
OUTPUTFILE = output

PROGRAM = main
EXEC = exec

$(EXEC).out:
	$(CC) $(CFLAGS) $(PROGRAM).cu -o $(EXEC).out

run:
	./$(EXEC).out $(INITFILE) $(INPUTFILE) $(OUTPUTFILE) 

clean:
	rm -rf *~ *.out *.o *# *.log

all: $(EXEC).out run


