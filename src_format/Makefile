
format: format.o vocab.o
	nvcc -o format format.o vocab.o

vocab.o:../src_culda/vocab.cpp ../src_culda/vocab.h
	nvcc -c ../src_culda/vocab.cpp -o vocab.o

format.o:format.cpp  ../src_culda/vocab.h data_chunk.h
	nvcc -c format.cpp -o format.o

clean:
	rm -f *.o format