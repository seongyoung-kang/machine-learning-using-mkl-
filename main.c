#include <stdio.h>
#include <stdlib.h>
#include "network.h"

#define SGD_CONF_PATH "./network_configuration/sgd.conf"

int main(int argc, char **argv)
{
	/*
		TODO(casionwoo) :

		Generate network struct
		send the configuration file_path to network object
	*/
	struct network *sgd;
	if ((sgd = (struct network *) malloc(sizeof(struct network))) == NULL) {
		printf("sgd struct generate failed\n");
		exit(1);
	}

	// Initialze from configuration file
	initializer(sgd, SGD_CONF_PATH);

	// read and fill up network input later
	reader(sgd);

	// run the training
	update(sgd);

    printf("program end\n");

    return 0;
}
