#include <stdio.h>
#include <stdlib.h>
#include "network.h"

#define CONF_FILE   "./network_configuration/sgd0.conf"

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
        threads[i-1] = atoi(argv[i]);

    conf_str = read_conf_file(CONF_FILE);

    net = (struct network *) malloc(sizeof(struct network));

    init(net, conf_str);

    reader(net);

    train(net, (void *) threads);

//    predict(net);

    report(net, (void *) threads);

    free(net);

    return 0;
}

char *read_conf_file(char *conf_name)
{
	FILE *fp;
	long lSize;
	char *buffer;

	if ((fp = fopen ( conf_name , "rb" )) == NULL) {
		printf("%s fopen failed\n", conf_name);
		exit(1);
	}

	fseek( fp , 0L , SEEK_END);
	lSize = ftell( fp );
	rewind( fp );

	/* allocate memory for entire content */
	if ((buffer = (char *) calloc(1, lSize+1)) == NULL) {
		fclose(fp);
		printf("buffer memory alloc fails\n");
		exit(1);
	}

	/* copy the file into the buffer */
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
