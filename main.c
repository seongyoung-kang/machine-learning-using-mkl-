#include <stdio.h>
#include <stdlib.h>
#include "network.h"

#define SGD_CONF_PATH "./network_configuration/sgd.conf"

int main(int argc, char **argv)
{
	struct network *sgd;
	if ((sgd = (struct network *) malloc(sizeof(struct network))) == NULL) {
		printf("sgd struct generate failed\n");
		exit(1);
	}

	run(sgd, SGD_CONF_PATH);

	// TODO(casionwoo) : report the result 
	// 1) print
	// 2) file

    return 0;
}
