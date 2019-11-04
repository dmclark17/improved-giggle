#include <cstdio>
#include <string>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Program takes number of matrices and their size\n");
        return 0;
    }
    int num, size;
    num = atoi(argv[1]);
    size = atoi(argv[2]);

    printf("Generating %d matrices of size %d\n", num, size);

}
