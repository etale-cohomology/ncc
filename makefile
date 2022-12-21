all: ncc

ncc: ncc.c  makefile ${mathisart}/mathisart4.h
	t tcc   ncc.c -o ncc
# t tcc   ncc.c -o ncc
# t gcc-8 ncc.c -o ncc
# t gcc-8 ncc.c -o ncc  $cflags $cnopie $cfast

clean:
	rm -f ncc.c
