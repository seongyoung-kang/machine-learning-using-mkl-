#include "network.h"
#include <stdio.h>
#include <stdlib.h>

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist/mnist.h"
#include "timeutils.h"
#include <math.h>
#include "mkl.h"

#include <omp.h>

#define MAX_CPU         256

void feedforward(struct network *net, int thread, int mode);
void back_pass(struct network *net, int thread1, int thread2, int mode);
void backpropagation(struct network *net, int thread1, int thread2, int mode);

double calculate_feedforward(struct network *net, int thread,int mode);
double calculate_back_pass_1(struct network *net, int thread,int mode);
double calculate_back_pass_2(struct network *net, int thread,int mode);
double calculate_backpropagation_2(struct network *net, int thread,int mode);
double calculate_backpropagation_1(struct network *net, int thread,int mode);

static double randn(void);
static double sigmoid(double z);
static double sigmoid_prime(double z);

/* Init network struct from configuration file */
void init(struct network *net, char *conf_str)
{
	int i,j,k;
	int before_ac_weights = 0;
	int before_ac_neurals = 0;

    timeutils *feedforward = &net->t_feedforward; //timeutils 는 net 에 있는 t_feedforward 의 주소값을 가진다.
    timeutils *back_pass = &net->t_back_pass;
    timeutils *backpropagation = &net->t_backpropagation;

	net->best_recog = 0.0;
    TIMER_INIT(feedforward); //시간 초기화
    TIMER_INIT(back_pass);
    TIMER_INIT(backpropagation);

    /*json 을 이용해서 값을 불러와서 net에 값을 넣어준다 */

    net->tokens = json_parsing(conf_str, &net->nr_tokens);
	net->num_layer = atoi((char *) parse_value(net->tokens, conf_str, "num_layer", net->nr_tokens));
	net->layer_size = (int *) parse_value(net->tokens, conf_str, "layer_size", net->nr_tokens);
	net->learning_rate = strtod((char *) parse_value(net->tokens, conf_str, "learning_rate", net->nr_tokens), NULL);
	net->mini_batch_size = atoi((char *) parse_value(net->tokens, conf_str, "mini_batch_size", net->nr_tokens));
	net->epoch = atoi((char *) parse_value(net->tokens, conf_str, "epoch", net->nr_tokens));

	net->ac_weight = (int *) malloc(sizeof(double) * net->num_layer);
	net->ac_neuron = (int *) malloc(sizeof(double) * net->num_layer);

	net->train_q_name = (char *) parse_value(net->tokens, conf_str, "train_q", net->nr_tokens); //network.h 헤더파일에 있는 부분
	net->train_a_name = (char *) parse_value(net->tokens, conf_str, "train_a", net->nr_tokens);
	net->test_q_name = (char *) parse_value(net->tokens, conf_str, "test_q", net->nr_tokens);
	net->test_a_name = (char *) parse_value(net->tokens, conf_str, "test_a", net->nr_tokens);
	net->report_file = (char *) parse_value(net->tokens, conf_str, "report_file", net->nr_tokens);


	for (i = 0; i < net->num_layer; i++) {
		net->ac_neuron[i] = net->layer_size[i] + before_ac_neurals;//ac_neuron은 여태 누적한 neuron갯수..
		before_ac_neurals = net->ac_neuron[i];

		if (i == net->num_layer-1)
			continue;

		net->ac_weight[i] = net->layer_size[i] * net->layer_size[i+1] + before_ac_weights; //ac_weight는 여태 누적한 weight 의 갯수..
		before_ac_weights = net->ac_weight[i]; 
	}

	net->neuron = (double *) malloc(sizeof(double) * net->mini_batch_size * TOTAL_NEURONS(net)); //neuron 배열의 크기는 minibatch_size * 총 뉴련의 숫자
	net->zs = (double *) malloc(sizeof(double) * net->mini_batch_size * TOTAL_NEURONS(net));
	net->error =  (double *) malloc(sizeof(double) * net->mini_batch_size * TOTAL_NEURONS(net));
	net->bias = (double *) malloc(sizeof(double) * TOTAL_NEURONS(net));
	net->weight = (double *) malloc(sizeof(double) * TOTAL_WEIGHTS(net));

	// init weight with bias with random values
	for (i = 0; i < TOTAL_WEIGHTS(net); i++) {
        net->weight[i] = randn();
	}

	for (i = 0; i < TOTAL_NEURONS(net); i++) {
        net->bias[i] = randn();
	}

	free(conf_str);
}

/* Read and fillup the first layer of neurons */
void reader(struct network *net)
{
	int i, j, k, l;
	mnist_data *train_data, *test_data;
	int first_layer_size = AC_NEURONS(net, 0);
	int last_layer_size = net->layer_size[net->num_layer-1];

	// Reading train data
	if (mnist_load(net->train_q_name, net->train_a_name, &train_data, &net->nr_train_data) < 0)
		printf("Mnist train data reading error occured\n");

	net->train_q = (double *) malloc(net->layer_size[0] * net->nr_train_data * sizeof(double));
	net->train_a = (int *) calloc(net->nr_train_data, sizeof(int));

	// copy train imgae&label
	for (i = 0; i < net->nr_train_data; i++) {
		for (j = 0; j < first_layer_size; j++) {
			DATA_TRAIN_Q(net, i, j) = train_data[i].data[j/28][j%28];       //train_q[] =  train_data[data number].data[28][28 ] train_q배열은 28*28*데이터 수 만큼의 크기임
         }
        DATA_TRAIN_A(net, i) = train_data[i].label;                         //train_a[] =  답안 배열 크기는 데이터 수 만큼의 크기 입니다.
	}

	// Reading test data
	if (mnist_load(net->test_q_name, net->test_a_name, &test_data, &net->nr_test_data) < 0)
		printf("Mnist test data reading error occured\n");

	net->test_q = (double *) malloc(net->layer_size[0] * net->nr_test_data * sizeof(double));
	net->test_a = (int *) calloc(net->nr_test_data, sizeof(int));

	// copy test imgae&label
	for (i = 0; i < net->nr_test_data; i++) {
		for (j = 0; j < first_layer_size; j++) {
			DATA_TEST_Q(net, i, j) = test_data[i].data[j/28][j%28];
		}

        DATA_TEST_A(net, i) = test_data[i].label;
	}

	free(train_data);
	free(test_data);
}

// run the training
void train(struct network *net, void *threads, void *modes)
{
	int i, j, k, l;
	int nr_train = net->nr_train_data;
	int nr_loop = (int)(net->nr_train_data/net->mini_batch_size);   //전체데이터를 미니배치 사이즈 만큼 나눈 수 입니다.(업데이트 할 숫자)
	int first_layer_size = AC_NEURONS(net, 0);                      //말 그대로 첫번째 layer size
	int last_layer_size = net->layer_size[net->num_layer-1];        //맨마지막에서 전단계 layer size
	int recog = 0;
    int *thread = (int *) threads;
	int *mode = (int *)modes;

	
	 for (i = 0; i < net->epoch; i++) {
		for (j = 0; j < nr_loop; j++) {                                 //j는 업데이트 하는 번수 (전체데이터를  mini batch로 나눈 값)

			// copy input and output for SGD
			for (k = 0; k < net->mini_batch_size; k++) {                   //k는데이터 번호를 뜻합니다, mini batch 사이즈 전까지 증가합니다
                int s_index = (int) rand()%nr_train;
				// copy input to first layer of neuron array
				for (l = 0; l < first_layer_size; l++)                    //l은 28*28 까지 증가합니다
					NEURON(net, 0, k, l) = DATA_TRAIN_Q(net, s_index, l); //s_index 번째 데이터를 가져옵니다 그것을 net->neuron[net->layer_size[0]*(k) + (l)] 에 넣습니다.
                                                                        //즉 neuron 배열에 차곡차곡 랜덤한 인풋값을 넣습니다.
               for (l = 0; l < last_layer_size; l++)
                    ERROR(net, net->num_layer-1, k, l) = 0.0;
				// copy output to error array
				ERROR(net, net->num_layer-1, k, DATA_TRAIN_A(net, s_index)) = 1.0; //error 배열에 0또는 1의값 넣습니다.
			}
            // feedforward + back_pass      mini_batch size 만큼 다하고 함수들 실행
            feedforward(net, thread[0],mode[0]);
            back_pass(net, thread[1], thread[2],mode[1]);
            backpropagation(net, thread[3], thread[4],mode[2]);
		}
		// test per every epoch
		recog = predict(net);
		if (recog > net->best_recog)
			net->best_recog = recog;
		printf("%dth epoch %d / %d\n", i, recog, net->nr_test_data);
	}
}

void feedforward(struct network *net, int thread, int mode)
{
	int i, j, k, l, m;
	double sum = 0.0;
    timeutils *feedforward = &net->t_feedforward;

	// feedforward
	START_TIME(feedforward);
    sum = 0.0;
    int nr_chunk = thread;
    int chunk_size = (int) (net->mini_batch_size/nr_chunk); //mini_batch size를 쓰래드 갯수만큼  나눈것

if(0)   /* OpenMP : without collapse */
{
// omp_set_nested(1); // without this line is fater than with.
    for (i = 0; i < net->num_layer-1; i++)  //layer 갯수에서 하나뺀 즉, 실질적으로 한번씩 값의 전파가 일어나는 횟수
    {
        for (j = 0; j < nr_chunk; j++)
		{ //설정해준 쓰래드 갯수만큼 (예를 들어 쓰래드가  100개 , mini_batch가 500 이면 100만큼)
			#pragma omp parallel for num_threads(thread) private(m,k,l)
			for (m = 0; m < chunk_size; m++) // 0~5 까지
             {
//                printf("m(%d) : %d/%d th thread\n", m,omp_get_thread_num(),omp_get_num_threads());

               #pragma omp parallel for num_threads(MAX_CPU/chunk_size) private(k, l) // k,l 을 private하게 input 하나를 잡고 돌리는거임.
               for (k = 0; k < net->layer_size[i+1]; k++)
                     {
//                        printf("m(%d), k(%d) : %d/%d th thread\n", m, k, omp_get_thread_num(),omp_get_num_threads()); // 현재쓰래드 / 전체 쓰래드

                        #pragma omp simd reduction(+:sum)
                         for (l = 0; l < net->layer_size[i]; l++)
                             {
                                 sum = sum + NEURON(net, i, j*chunk_size+m, l) * WEIGHT(net, i, l, k);

                             // NEURON(net, i, j, k) = ith layer, jth mini_batch, kth node
                             }


                        ZS(net, i+1, j*chunk_size+m, k) = sum + BIAS(net, i+1, k);
                        NEURON(net, i+1, j*chunk_size+m, k) = sigmoid(ZS(net, i+1, j*chunk_size+m, k));
                        sum = 0.0;
                    }
            }
        }
	}
}

/*    for (i = 0; i < net->num_layer-1; i++) {
        for (j = 0; j < nr_chunk; j++) {
            #pragma omp parallel for num_threads(100) private(m, k, l) collapse(2)
            for (m = 0; m < chunk_size; m++) {
                for (k = 0; k < net->layer_size[i+1]; k++) {
                    #pragma omp simd reduction(+:sum)
                    for (l = 0; l < net->layer_size[i]; l++) {
                        sum = sum + NEURON(net, i, j*chunk_size+m, l) * WEIGHT(net, i, l, k);
                    }

                    ZS(net, i+1, j*chunk_size+m, k) = sum + BIAS(net, i+1, k);
                    NEURON(net, i+1, j*chunk_size+m, k) = sigmoid(ZS(net, i+1, j*chunk_size+m, k));
                    sum = 0.0;
                }
            }
        }
    }
*/
else
{
    double *tmp, *tmp_bias;

    for (i = 0; i < net->num_layer-1; i++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, net->mini_batch_size, net->layer_size[i+1], net->layer_size[i], 1.0, (const double *)&NEURON(net, i, 0, 0),net->layer_size[i], (const double *)&WEIGHT(net, i, 0, 0), net->layer_size[i+1], 0.0,&NEURON(net, i+1, 0, 0), net->layer_size[i+1]); //weight 와 입력값을 곱해서 배열에 저장합니다.

        tmp      = malloc(sizeof(double) * net->mini_batch_size);
        tmp_bias = malloc(sizeof(double) * net->layer_size[i+1] * net->mini_batch_size);
        for (j = 0; j < net->mini_batch_size; j++)
            tmp[j] = 1.0;

        cblas_dger(CblasRowMajor, net->mini_batch_size, net->layer_size[i+1],
                        1.0, (const double *)tmp, 1, (const double *)&BIAS(net, i, 0),
                        1, tmp_bias, net->layer_size[i+1]); // tmp 라는 임시 배열을 만들어서 백터 두개를 합쳐서 행렬로 만듭니다. 그리고 그것을 tmp_bias에 저장합니다

        vdAdd(net->layer_size[i+1] * net->mini_batch_size, tmp_bias, &NEURON(net, i+1, 0, 0), &ZS(net, i+1, 0, 0)); //그리고 bias랑 값을 더한것을 zs에 저장합니다
        for (j = 0; j < net->mini_batch_size; j++)
            for (k = 0; k < net->layer_size[i+1]; k++)
                NEURON(net, i+1, j, k) = sigmoid(ZS(net, i+1, j, k)); //zs에  sigmoid를 취한 값을 그다음 뉴런에 저장합니다!!
    }
}
	END_TIME(feedforward);
}

void back_pass(struct network *net, int thread1, int thread2, int mode)
{
	int i, j, k, l;
	double sum = 0.0;
    timeutils *back_pass = &net->t_back_pass;

	START_TIME(back_pass);
if(mode)
{
// calculate delta
#pragma omp parallel for num_threads(thread1) private(i, j) collapse(2)
	for (i = 0; i < net->mini_batch_size; i++) {
		for (j = 0; j < net->layer_size[net->num_layer-1]; j++) {
			//	calculate delta in last output layer
			ERROR(net, net->num_layer-1, i, j) =
			(NEURON(net, net->num_layer-1, i, j)-ERROR(net, net->num_layer-1, i, j)) *
			sigmoid_prime(ZS(net, net->num_layer-1, i, j));
		}
	}

	sum = 0.0;
	for (i = net->num_layer-2; i > 0; i--) {
#pragma omp parallel for num_threads(thread2) private(j, k, l) reduction(+:sum) collapse(2)
		for (j = 0; j < net->mini_batch_size; j++) {
			for (k = 0; k < net->layer_size[i]; k++) {
				for (l = 0; l < net->layer_size[i+1]; l++) {
					//	calculate delta from before layer
					sum = sum + ERROR(net, i+1, j, l) * WEIGHT(net, i, k, l);
				}
				ERROR(net, i, j, k) = sum * sigmoid_prime(ZS(net, i, j, k));
				sum = 0.0;
			}
		}
	}
}
else
{
	double * temp1;//neuron - error
    double * temp2;//sigmoid zs
    double * temp_error;

    temp1 = (double*)malloc(sizeof(double) * net->mini_batch_size * net->layer_size[net->num_layer-1]);
    temp2 = (double*)malloc(sizeof(double) * net->mini_batch_size * net->layer_size[net->num_layer-1]);

    // neuron - error
    vdSub(net->layer_size[net->num_layer-1]*net->mini_batch_size,&NEURON(net, net->num_layer-1, 0, 0),&ERROR(net, net->num_layer-1, 0, 0),temp1);

    //sigmoid zs
    #pragma omp parallel for num_threads(thread1)
	    for (i = 0; i < net->mini_batch_size*net->layer_size[net->num_layer-1]; i++)
            {
		         temp2[i]=sigmoid_prime(ZS(net, net->num_layer-1, 0, i));
	    	}

    //temp1 * temp2 (when this loop is end  first delta is done!!)
    vdMul(net->layer_size[net->num_layer-1]*net->mini_batch_size,temp1,temp2,&ERROR(net, net->num_layer-1, 0, 0));

    //caculrate delta to using backpropagation algorithm
    for (i = net->num_layer-2; i > 0; i--)
    {
		for (j = 0; j < net->mini_batch_size; j++)
        {
            //temp_error = weight * past_error
            temp_error = (double*)malloc(sizeof(double)*net->layer_size[i]);

            //calculate temp_error
            cblas_dgemv (CblasRowMajor, CblasNoTrans,  net->layer_size[i], net->layer_size[i+1], 1.0,(const double *)&WEIGHT(net, i, 0, 0), net->layer_size[i+1],(const double *)&ERROR(net,i+1, j, 0),1 ,0.0 , temp_error , 1);

            //calculate delta = past error * weight * sigmoidprime(zs)
            #pragma omp parallel for num_threads(thread1)
            for(k=0;k<net->layer_size[i];k++)
            {
                ERROR(net, i, j, k) = temp_error[k]*sigmoid_prime(ZS(net, i, j, k));
            }

        }
    }
}
	END_TIME(back_pass);
}

/* Operation like backpropagation */
void backpropagation(struct network *net, int thread1, int thread2,int mode)
{
	int i, j, k, l;
    timeutils *backpropagation = &net->t_backpropagation;
	double eta = net->learning_rate;
	double mini = (double) net->mini_batch_size;

	START_TIME(backpropagation);
	// update bias
#pragma omp parallel for num_threads(thread1) private(i, j, k) collapse(2)
	for (i = 1; i < net->num_layer; i++) {
		for (j = 0; j < net->layer_size[i]; j++) {
            #pragma omp simd
			for (k = 0; k < net->mini_batch_size; k++) {
                BIAS(net, i, j) -= (eta/mini)*ERROR(net, i, k, j);
			}
		}
	}

if(mode)
{	// update weight
	for (i = 0; i < net->num_layer-1; i++) {
#pragma omp parallel for num_threads(thread2) private(j, k, l) collapse(2)
		for (j = 0; j < net->layer_size[i]; j++) {
			for (k = 0; k < net->layer_size[i+1]; k++) {
                #pragma omp simd
				for (l = 0; l < net->mini_batch_size; l++) {
					//	calculate delta from before layer
                    WEIGHT(net, i, j, k) -= (eta/mini)*(NEURON(net, i, l, j) * ERROR(net, i+1, l, k));
				}
			}
		}
	}
}
	// update weight
else
{
	for (i = 0; i < net->num_layer-1; i++)
	{
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,net->layer_size[i], net->layer_size[i+1],net->mini_batch_size, -(eta/mini), (const double *)&NEURON(net, i, 0, 0),net->layer_size[i], (const double *)&ERROR(net, i+1, 0, 0), net->layer_size[i+1], 1.0,&WEIGHT(net, i, 0, 0), net->layer_size[i+1]);
	}
}
	END_TIME(backpropagation);
}

double sigmoid(double z)
{
	return (1/(1 + exp(-z)));
}

double sigmoid_prime(double z)
{
	return sigmoid(z)*(1-sigmoid(z));
}

int predict(struct network *net)
{
	int nr_true = 0;

	int i, j, k, l;
	double sum = 0.0;
	int nr_loop = (int)(net->nr_test_data);
	int first_layer_size = AC_NEURONS(net, 0);
	int last_layer_size = net->layer_size[net->num_layer-1];

	for (i = 0; i < nr_loop; i++) {
		// copy input to first layer of neuron array
		for (j = 0; j < first_layer_size; j++) {
			NEURON(net, 0, 0, j) = DATA_TEST_Q(net, i, j);
		}

		//feedforward
        sum = 0.0;
		for (j = 0; j < net->num_layer-1; j++) {
#pragma omp parallel for num_threads(100) private(k, l) reduction(+:sum)
			for (k = 0; k < net->layer_size[j+1]; k++) {
				for (l = 0; l < net->layer_size[j]; l++) {
					sum = sum + NEURON(net, j, 0, l) * WEIGHT(net, j, l, k);
				}

				ZS(net, j+1, 0, k) = sum + BIAS(net, j+1, k);
				NEURON(net, j+1, 0, k) = sigmoid(ZS(net, j+1, 0, k));
				sum = 0.0;
			}
		}

		double max = NEURON(net, net->num_layer-1, 0, 0);
		int max_idx = 0;

		for (j = 0; j < last_layer_size; j++) {
			if (NEURON(net, net->num_layer-1, 0, j) > max) {
				max = NEURON(net, net->num_layer-1, 0, j);
				max_idx = j;
			}
		}

		if (DATA_TEST_A(net, i) == max_idx)
			nr_true ++;
	}

	return nr_true;
}

void report(struct network *net, void *threads,void *modes)
{
    int *thread = (int *) threads;
	int *mode = (int *) modes;

    timeutils *feedforward = &net->t_feedforward;
    timeutils *back_pass = &net->t_back_pass;
    timeutils *backpropagation = &net->t_backpropagation;
    timeutils t;
    timeutils *total = &t;

    TIMER_INIT(total);

	char *modeid[2] = {"MKL","OpenMP"};
	int i = 0;
	FILE *f = fopen(net->report_file, "a+");
	FILE *f_json = fopen("./result/dump_json", "a+");
	if (f == NULL || f_json == NULL) {
		printf("%s open failed\n", net->report_file);printf("%s open failed\n", "dump_json");
		exit(1);
	}
	fprintf( f, "\n=======================REPORT=======================\n");
//	fprintf( f, "==================HYPER-PARAMETERS================== \n");
//	fprintf( f, "mini_batch_size : %d\n", net->mini_batch_size);
//	fprintf( f, "num_layer : %d\n", net->num_layer);
//	fprintf( f, "layers : [");
//	for (i = 0; i < net->num_layer; i++) {
//		fprintf( f, "%d ", net->layer_size[i]);
//	}
//	fprintf( f, "]\n");
	fprintf( f, "epoch : %d\n", net->epoch);
	fprintf( f, "learning_rate : %f\n", net->learning_rate);
	fprintf( f, "recognization rate : %d/%d\n", net->best_recog, net->nr_test_data);
	fprintf( f, "=======================THREADS======================\n");
	fprintf( f, "feedforward thread : %d\n", thread[0]);
	fprintf( f, "back_pass thread1 : %d\n", thread[1]);
	fprintf( f, "back_pass thread2 : %d\n", thread[2]);
	fprintf( f, "backpropagation thread1 : %d\n", thread[3]);
	fprintf( f, "backpropagation thread2 : %d\n", thread[4]);
	fprintf( f, "========================MODE========================\n");
	fprintf( f, "feedforward mode (error) : %s\n", modeid[0]);
	fprintf( f, "back_pass mode : %s\n", modeid[mode[1]]);
	fprintf( f, "backpropagation mode : %s\n", modeid[mode[2]]);
	fprintf( f, "========================TIME========================\n");
	fprintf( f, "feedforward : %ld.%d sec\n", TOTAL_SEC_TIME(feedforward), TOTAL_SEC_UTIME(feedforward));
	fprintf( f, "back_pass : %ld.%d sec\n", TOTAL_SEC_TIME(back_pass), TOTAL_SEC_UTIME(back_pass));
	fprintf( f, "backpropagation : %ld.%d sec\n", TOTAL_SEC_TIME(backpropagation), TOTAL_SEC_UTIME(backpropagation));

    TIMER_ADD(feedforward, total);
    TIMER_ADD(back_pass, total);
    TIMER_ADD(backpropagation, total);
	fprintf( f, "total : %ld.%d sec\n", TOTAL_SEC_TIME(total), TOTAL_SEC_UTIME(total));

    fprintf( f_json, "{\n");
    fprintf( f_json, "\"feedforward_thread\":%d,\n",thread[0]);
    fprintf( f_json, "\"back_pass_thread1\":%d,\n",thread[1]);
    fprintf( f_json, "\"back_pass_thread2\":%d,\n",thread[2]);
    fprintf( f_json, "\"backpropagation_thread1\":%d,\n",thread[3]);
    fprintf( f_json, "\"backpropagation_thread2\":%d,\n",thread[4]);
	fprintf( f_json, "\"feedforward_time\": %ld.%d,\n", TOTAL_SEC_TIME(feedforward), TOTAL_SEC_UTIME(feedforward));
	fprintf( f_json, "\"back_pass_time\": %ld.%d,\n", TOTAL_SEC_TIME(back_pass), TOTAL_SEC_UTIME(back_pass));
	fprintf( f_json, "\"backpropagation\": %ld.%d,\n", TOTAL_SEC_TIME(backpropagation), TOTAL_SEC_UTIME(backpropagation));
	fprintf( f_json, "\"total_time\": %ld.%d\n", TOTAL_SEC_TIME(total), TOTAL_SEC_UTIME(total));
    fprintf( f_json, "},");

	fclose(f);
	fclose(f_json);
}

double randn(void)
{
    double v1, v2, s;

    do {
        v1 =  2 * ((double) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
        v2 =  2 * ((double) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
        s = v1 * v1 + v2 * v2;
    } while (s >= 1 || s == 0);

    s = sqrt( (-2 * log(s)) / s );

    return v1 * s;
}



void setting(struct network *net, void *threads, void* modes, int maxthreads)
{
	int i, j, k, l;
	int nr_train = net->nr_train_data;
	int first_layer_size = AC_NEURONS(net, 0);
	int last_layer_size = net->layer_size[net->num_layer-1];
    int *thread = (int *) threads;
	int *mode = (int *)modes;

	int s_index = (int) rand()%nr_train;
	
	double temp = 0;
	double feedforward_t = -1;
    double back_pass_t1 = -1;
	double back_pass_t2 = -1;
	double backpropagation_t1 = -1;
	double backpropagation_t2 = -1;
	
	// copy input to first layer of neuron array
	for (l = 0; l < first_layer_size; l++)
		NEURON(net, 0, k, l) = DATA_TRAIN_Q(net, s_index, l);


	for (k = 0; k < net->mini_batch_size; k++) 
	{

        for (l = 0; l < last_layer_size; l++)
             ERROR(net, net->num_layer-1, k, l) = 0.0;
				
		// copy output to error array
		ERROR(net, net->num_layer-1, k, DATA_TRAIN_A(net, s_index)) = 1.0;

	}
    
	// setting threads      mini_batch size 만큼 다하고 함수들 실행
	
	timeutils *tfeedforward = &net->t_feedforward; //timeutils 는 net 에 있는 t_feedforward 의 주소값을 가진다.
	timeutils *tback_pass = &net->t_back_pass;
	timeutils *tbackpropagation = &net->t_backpropagation;
    
	TIMER_INIT(tfeedforward); //시간 초기화
    TIMER_INIT(tback_pass);
    TIMER_INIT(tbackpropagation);
	
	for(i=1;i<maxthreads;i+= 5)
	{
		temp = calculate_feedforward(net,i,1);
	   if(feedforward_t == -1 || feedforward_t >  temp)
	   {
		   feedforward_t = temp;
		   thread[0] = i;
	   }
	   
	   temp = calculate_back_pass_1(net,i,1);
	   if(back_pass_t1 == -1 || back_pass_t1 > temp)
	   {
		   back_pass_t1 = temp;
		   thread[1] = i;
	   }

	   temp = calculate_back_pass_2(net,i,1);
	   if(back_pass_t2 == -1 || back_pass_t2 > temp)
	   {
		   back_pass_t2 = temp;
		   thread[2] = i;
	   }
	   temp = calculate_backpropagation_1(net,i,1);
	   if(backpropagation_t1 == -1 || backpropagation_t1 > temp)
	   {
		   backpropagation_t1 = temp;
		   thread[3] = i;
	   }
		
	   temp = calculate_backpropagation_2(net,i,1);
	   if(backpropagation_t2 == -1 || backpropagation_t2 > temp)
	   {
		   backpropagation_t2 = temp;
		   thread[4] = i;
	   }
	}
	printf(" \n here %f  \n",(double)DIFF_TIME(tfeedforward));

	    feedforward(net,thread[0],0);
		temp = (double)DIFF_TIME(tfeedforward);
		feedforward(net,thread[0],1);
		if(temp < (double)DIFF_TIME(tfeedforward))
			mode[0]=0;
		
		back_pass(net,thread[1],thread[2],0);
		temp = (double)DIFF_TIME(tback_pass);
		back_pass(net,thread[1],thread[2],1);
		if(temp < (double)DIFF_TIME(tback_pass))
			mode[1]=0;
		
		backpropagation(net,thread[3],thread[4],0);
		temp = (double)DIFF_TIME(tbackpropagation);
		backpropagation(net,thread[3],thread[4],1);
		if(temp < (double)DIFF_TIME(tbackpropagation))
			mode[2] = 0;

	for(i=0;i<5;i++)
	{
		printf(" %d thread = %d\n",i,thread[i]);
	}
	printf("\n");
	for(i=0;i<3;i++)
	{
		printf(" %d mode = %d\n",i,mode[i]);
	}
	printf("\n");

	for (i = 0; i < TOTAL_WEIGHTS(net); i++) {
        net->weight[i] = randn();
	}

	for (i = 0; i < TOTAL_NEURONS(net); i++) {
        net->bias[i] = randn();
	}

    TIMER_INIT(tfeedforward); //시간 초기화
    TIMER_INIT(tback_pass);
    TIMER_INIT(tbackpropagation);
}

double calculate_feedforward(struct network *net, int thread,int mode)
{
		timeutils *tfeedforward = &net->t_feedforward; //timeutils 는 net 에 있는 t_feedforward 의 주소값을 가진다.
	    START_TIME(tfeedforward);
		feedforward(net, thread,mode);
		END_TIME(tfeedforward);
		return (double)DIFF_TIME(tfeedforward);
}
double calculate_back_pass_1(struct network *net, int thread,int mode)
{
   	 	timeutils *tback_pass = &net->t_back_pass;
	    START_TIME(tback_pass);
	    back_pass(net, thread,1,mode);
		END_TIME(tback_pass);
	    return(double)DIFF_TIME(tback_pass);
}

double calculate_back_pass_2(struct network *net, int thread,int mode)
{
   	 	timeutils *tback_pass = &net->t_back_pass;
	    START_TIME(tback_pass);
	    back_pass(net, 1,thread,mode);
	    END_TIME(tback_pass);
	    return (double)DIFF_TIME(tback_pass);
}

double calculate_backpropagation_1(struct network *net, int thread,int mode)
{
    	timeutils *tbackpropagation = &net->t_backpropagation;
	    START_TIME(tbackpropagation);
	    backpropagation(net, thread,1,mode);
	    END_TIME(tbackpropagation);
	    return (double)DIFF_TIME(tbackpropagation);
}

double calculate_backpropagation_2(struct network *net, int thread,int mode)
{
    	timeutils *tbackpropagation = &net->t_backpropagation;
	    START_TIME(tbackpropagation);
	    backpropagation(net,1,thread,mode);
	    END_TIME(tbackpropagation);
	    return (double)DIFF_TIME(tbackpropagation);
}

