#include <stdio.h>
#include <stdlib.h>
#include "network.h"

#define CONF_FILE   "./network_configuration/sgd0.conf" //mnist 원본 데이터

char *read_conf_file(char *conf_name);
void params_checker(int argc);

int main(int argc, char **argv)
{
    int i;
	int recog = 0;
	char *conf_str;
	struct network *net;
    int *threads;

    params_checker(argc);

    threads = (int *) malloc(sizeof(int) * argc-1);

    for (i = 1; i < argc; i++)
        threads[i-1] = atoi(argv[i]); //threads 배열은 쓰래드의 숫자를 각각 저장합니다.

    conf_str = read_conf_file(CONF_FILE); //conf_str에 json파일내용을 집어 넣습니다.

    net = (struct network *) malloc(sizeof(struct network)); //network 데이터를 할당합니다

    init(net, conf_str); //네트워크구조체와 , json 형태의 배열을 보내줍니다. 네트워크 구조체 멤버들의 메모리크기 할당& 초기화

    reader(net); //train , test 에 input값을 넣어주는 역할을 합니다.

    train(net, (void *) threads); //net에 있는 neuron 과 error에 적절한 값을 다 넣어준후 backpropagation train 을 합니다

//    predict(net);

    report(net, (void *) threads);

    free(net);

    return 0;
}

char *read_conf_file(char *conf_name) //json 형태의 파일을 읽습니다.
{
	FILE *fp;
	long lSize;
	char *buffer; //반환될 값입니다

	if ((fp = fopen ( conf_name , "rb" )) == NULL) { //fp는json 파일을 가르킵니다.
		printf("%s fopen failed\n", conf_name);
		exit(1);
	}

	fseek( fp , 0L , SEEK_END);
	lSize = ftell( fp );//파일 크기 알아보기 위해서 씀.
	rewind( fp );

	/* allocate memory for entire content */ //버퍼에 json 파일 만큼 크기배열을 할당합니다.
	if ((buffer = (char *) calloc(1, lSize+1)) == NULL) {
		fclose(fp);
		printf("buffer memory alloc fails\n");
		exit(1);
	}

	/* copy the file into the buffer 버퍼에 json파일의 내용을 넣습니다 */
	if( 1!=fread( buffer , lSize, 1 , fp) ) {
		fclose(fp);
		free(buffer);
		printf("%s entire read fails\n", conf_name);
		exit(1);
	}

	return buffer;
}

void params_checker(int argc)
{
    if (argc < 6) {
        printf("please check your parameters\n");
        exit(1);
    }
}
