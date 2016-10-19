CC=gcc
ELF=mnist

ROOT_PATH= .

LIB_PATH = $(ROOT_PATH)/libs/

EVALUATOR_PATH= ./evaluator
FORWARDER_PATH= ./forwarder
LEARNER_PATH= ./learner
READER_PATH= ./reader

# OBJS += $(EVALUATOR_PATH)/
# OBJS += $(FORWARDER_PATH)/
# OBJS += $(LEARNER_PATH)/
# OBJS += $(READER_PATH)/

OBJS += main.o network.o jsmn.o
INCLUDE += $(LIB_PATH)

CFLAGS += -I$(INCLUDE)
CFLAGS += -g



all: ${ELF}

${ELF}: ${OBJS}
	$(CC) $(CFLAGS) $(OBJS) -o ${ELF}

network.o : ${ROOT_PATH}/network.c
	$(CC) $(CFLAGS) -c ${ROOT_PATH}/network.c

main.o : ${ROOT_PATH}/main.c
	$(CC) $(CFLAGS) -c ${ROOT_PATH}/main.c

jsmn.o : ${LIB_PATH}/jsmn/jsmn.c
	$(CC) $(CFLAGS) -c ${LIB_PATH}/jsmn/jsmn.c

clean:
	rm -rf ./*.o 
