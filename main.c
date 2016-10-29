#include <stdio.h>
#include <stdlib.h>
#include "network.h"

char *read_conf_file(char *conf_name);

char *CONF_FILES[] = {
	"./network_configuration/sgd0.conf",
	"./network_configuration/sgd1.conf",
	"./network_configuration/sgd2.conf",
	"./network_configuration/sgd3.conf",
	"./network_configuration/sgd4.conf",
	"./network_configuration/sgd5.conf",
	"./network_configuration/sgd6.conf",
	"./network_configuration/sgd7.conf",
	"./network_configuration/sgd8.conf",
	"./network_configuration/sgd9.conf",
};
int main(int argc, char **argv)
{
	int i = 0;
	char *conf_str;
	struct network *net;

    if ((net = (struct network *) malloc(sizeof(struct network))) == NULL) {
        printf("sgd struct generate failed\n");
        exit(1);
    }

    //TODO(casionwoo) : Read configuration from JSON
    conf_str = read_conf_file(CONF_FILES[0]);
	// Initialze from configuration file
    init(net, conf_str);
    //TODO(casionwoo) : Read Data set
    //TODO(casionwoo) : Call train() parameters : traing data set
    //TODO(casionwoo) : Call test()  parameters : test data set
    //TODO(casionwoo) : Call report() paramters : return value of test()

    run(net, CONF_FILES[i]);
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
