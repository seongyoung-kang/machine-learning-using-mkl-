#include <stdio.h>
#include <stdlib.h>
#include "network.h"
#include <timeutils.h>

#define SGD_CONF_PATH "./network_configuration/sgd.conf"

int main(int argc, char **argv)
{

	timeutils t_init, t_reader, t_update;

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
	START_TIME(t_init);
	initializer(sgd, SGD_CONF_PATH);
	END_TIME(t_init);



	// read and fill up network input later
	START_TIME(t_reader);
	reader(sgd);
	END_TIME(t_reader);

	// run the training
	START_TIME(t_update);
	update(sgd);
	END_TIME(t_update);

	// TODO(casionwoo) : report the result 
	// 1) print
	// 2) file
    printf("program end\n");

    return 0;
}
