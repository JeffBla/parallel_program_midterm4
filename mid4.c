#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int *A, *K, *local_A, *local_A_next;
int t, n, m, k_n, k_m;

void Get_Input(FILE *fp, int myid) {
    if (myid == 0) {
        fscanf(fp, "%d ", &t);
        fscanf(fp, "%d %d", &n, &m);
        A = malloc(sizeof(int) * n * m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                fscanf(fp, "%d", &A[i * m + j]);
            }
        }
        fscanf(fp, "%d %d", &k_n, &k_m);
        K = malloc(sizeof(int) * k_n * k_m);
        for (int i = 0; i < k_n; i++) {
            for (int j = 0; j < k_m; j++) {
                fscanf(fp, "%d", &K[i * k_m + j]);
            }
        }
    }
}

void Update_local_A(int myid, int numprocs, int *numElement, int buffer_row) {
    int num_used_procs = 0;
    while (num_used_procs < numprocs) {
        if (numElement[num_used_procs] == 0)
            break;
        num_used_procs++;
    }

    MPI_Status status;
    int downRank = (myid + 1) % num_used_procs;
    int upRank = (myid + num_used_procs - 1) % num_used_procs;
    MPI_Sendrecv(local_A + buffer_row * m, buffer_row * m, MPI_INT, upRank, 0,
                 local_A + numElement[myid] + buffer_row * m, buffer_row * m, MPI_INT, downRank, 0, MPI_COMM_WORLD, &status);

    MPI_Sendrecv(local_A + numElement[myid], buffer_row * m, MPI_INT, downRank, 0,
                 local_A, buffer_row * m, MPI_INT, upRank, 0, MPI_COMM_WORLD, &status);
}

void Distribution(int myid, int numprocs, int *local_n_arr, int *numElement, int *disp_arr, int buffer_row) {
    int nRow_buffer_row_based = n / buffer_row;
    int nRow_buffer_row_based_remain = n % buffer_row;

    int local_n_tmp = nRow_buffer_row_based / numprocs;
    int local_n_remain = nRow_buffer_row_based % numprocs;
    if (local_n_remain != 0) {
        int i = 0;
        for (; local_n_remain != 0; i++, local_n_remain--) {
            local_n_arr[i] = local_n_tmp * buffer_row + 1 * buffer_row;
            numElement[i] = local_n_arr[i] * m;
        }
        for (; i < numprocs; i++) {
            local_n_arr[i] = local_n_tmp * buffer_row;
            numElement[i] = local_n_arr[i] * m;
        }
    } else {
        for (int i = 0; i < numprocs; i++) {
            local_n_arr[i] = local_n_tmp * buffer_row;
            numElement[i] = local_n_arr[i] * m;
        }
    }
    if (nRow_buffer_row_based_remain != 0) {
        if (local_n_tmp == 0) {
            int nRow_remain_n = nRow_buffer_row_based_remain / local_n_remain;
            int nRow_remain__remain = nRow_buffer_row_based_remain % local_n_remain;

            int i = 0;
            for (; nRow_remain__remain != 0; i++, nRow_remain__remain--) {
                local_n_arr[i] += nRow_remain_n + 1;
                numElement[i] = local_n_arr[i] * m;
            }
            for (; i < numprocs; i++) {
                local_n_arr[i] += nRow_remain_n;
                numElement[i] = local_n_arr[i] * m;
            }
        } else {
            for (int i = 0; nRow_buffer_row_based_remain != 0; i++, nRow_buffer_row_based_remain--) {
                local_n_arr[i] += 1;
                numElement[i] = local_n_arr[i] * m;
            }
        }
    }
    disp_arr[0] = 0;
    for (int i = 1; i < numprocs; i++) {
        disp_arr[i] = disp_arr[i - 1] + numElement[i - 1];
    }

    local_A = malloc(sizeof(int) * (local_n_arr[myid] + buffer_row * 2) * m);
    local_A_next = malloc(sizeof(int) * (local_n_arr[myid] + buffer_row * 2) * m);
    MPI_Scatterv(A, numElement, disp_arr, MPI_INT,
                 local_A + buffer_row * m, numElement[myid], MPI_INT, 0, MPI_COMM_WORLD);

    if (numElement[myid] != 0)
        Update_local_A(myid, numprocs, numElement, buffer_row);

    if (myid != 0)
        K = malloc(sizeof(int) * k_n * k_m);
    MPI_Bcast(K, k_n * k_m, MPI_INT, 0, MPI_COMM_WORLD);
}

void PredictNext(int local_n, int buffer_row) {
    int k_row, k_col, a_row, a_col;
    int sum_a_ij;
    for (int i = buffer_row; i < local_n + buffer_row; i++) {
        for (int j = 0; j < m; j++) {
            sum_a_ij = 0;
            for (int d_i = -(k_n - 1) / 2; d_i <= (k_n - 1) / 2; d_i++) {
                for (int d_j = -(k_m - 1) / 2; d_j <= (k_m - 1) / 2; d_j++) {
                    k_row = (k_n - 1) / 2 + d_i;
                    k_col = (k_m - 1) / 2 + d_j;
                    a_row = i + d_i;
                    a_col = j + d_j;
                    if (a_col < 0) {
                        a_col += m;
                    } else if (a_col >= m) {
                        a_col -= m;
                    }
                    sum_a_ij += K[k_row * k_m + k_col] * local_A[a_row * m + a_col];
                }
            }
            local_A_next[i * m + j] = sum_a_ij / (k_n * k_m);
        }
    }
}

int main(int argc, char *argv[]) {
    int myid, numprocs;

    FILE *fp;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0) {
        char file_name[50];
        scanf("%s", file_name);
        fp = fopen(file_name, "r");
        Get_Input(fp, myid);
    }
    MPI_Bcast(&t, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k_m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int buffer_row = (k_n / 2);
    int numElement[numprocs];
    int local_n_arr[numprocs];
    int disp_arr[numprocs];
    Distribution(myid, numprocs, local_n_arr, numElement, disp_arr, buffer_row);

    for (int i = 0; i < t && numElement[myid] != 0; i++) {
        PredictNext(local_n_arr[myid], buffer_row);
/*        {
            for (int k = buffer_row; k < local_n_arr[myid] + buffer_row; k++) {
                for (int j = 0; j < m; j++) {
                    printf("%d ", local_A_next[k * m + j]);
                }
            }
            printf("\n------------------------------\n");
            fflush(stdout);
        } */
        memcpy(local_A, local_A_next, sizeof(int) * (local_n_arr[myid] + buffer_row * 2) * m);
        Update_local_A(myid, numprocs, numElement, buffer_row);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(local_A + buffer_row * m, numElement[myid], MPI_INT,
                A, numElement, disp_arr, MPI_INT, 0, MPI_COMM_WORLD);

     if (myid == 0) {
         for (int i = 0; i < n; i++) {
             for (int j = 0; j < m; j++) {
                 printf("%d ", A[i * m + j]);
             }
         }
     }

    MPI_Finalize();
    return 0;
}
