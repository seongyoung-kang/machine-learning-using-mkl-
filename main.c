#include <stdio.h>
#include <stdlib.h>
#include "network.h"

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
	struct network *sgd;

    if ((sgd = (struct network *) malloc(sizeof(struct network))) == NULL) {
        printf("sgd struct generate failed\n");
        exit(1);
    }

    run(sgd, CONF_FILES[i]);
    free(sgd);

    return 0;
}
