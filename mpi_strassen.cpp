#include <mpi.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <fstream>

using namespace std;

void Strassen(int n, int** matrix1, int** matrix2, int**& result, int rank, int size);

int** StrassenImp(int n, int** matrix1, int** matrix2);

int** CreateMatrix(int n);

void FillMatrix(int n, int**& matrix);

void DeleteMatrix(int n, int** matrix);

int** SliceMatrix(int n, int** matrix, int offseti, int offsetj);

int** SumMatrix(int n, int** matrix1, int** matrix2);

int** SubMatrix(int n, int** matrix1, int** matrix2);

int** MulMatrix(int n, int** matrix1, int** matrix2);

int** CombineMatrix(int m, int** c11, int** c12, int** c21, int** c22);

void PrintMatrix(int n, int** matrix);

int main(int argc, char* argv[])
{
    int rank;
    int size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    if (rank == 0)
    {
        cout << endl;
        cout << "Enter the dimensions of the matrix: ";
        cin >> n;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int** matrix1 = CreateMatrix(n);
    int** matrix2 = CreateMatrix(n);

    if (rank == 0)
    {
        FillMatrix(n, matrix1);
        FillMatrix(n, matrix2);
    }

    MPI_Bcast(&(matrix1[0][0]), n * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(matrix2[0][0]), n * n, MPI_INT, 0, MPI_COMM_WORLD);

    double startTime = MPI_Wtime();

    int** result;
    Strassen(n, matrix1, matrix2, result, rank, size);

    double endTime = MPI_Wtime();

    if (rank == 0)
    {
        cout << "\nParallel Strassen Runtime (MPI): ";
        cout << setprecision(5) << endTime - startTime << endl;
        cout << endl;
    }

    MPI_Finalize();

    return 0;
}

void Strassen(int n, int** matrix1, int** matrix2, int**& result, int rank, int size)
{
    if (n == 1)
    {
        result = CreateMatrix(1);
        result[0][0] = matrix1[0][0] * matrix2[0][0];
    }

    int m = n / 2;

    int** a = SliceMatrix(n, matrix1, 0, 0);
    int** b = SliceMatrix(n, matrix1, 0, m);
    int** c = SliceMatrix(n, matrix1, m, 0);
    int** d = SliceMatrix(n, matrix1, m, m);
    int** e = SliceMatrix(n, matrix2, 0, 0);
    int** f = SliceMatrix(n, matrix2, 0, m);
    int** g = SliceMatrix(n, matrix2, m, 0);
    int** h = SliceMatrix(n, matrix2, m, m);

    int** s1 = CreateMatrix(m);
    int** s2 = CreateMatrix(m);
    int** s3 = CreateMatrix(m);
    int** s4 = CreateMatrix(m);
    int** s5 = CreateMatrix(m);
    int** s6 = CreateMatrix(m);
    int** s7 = CreateMatrix(m);

    if (rank == 0)
    {
        MPI_Recv(&(s1[0][0]), m * m, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s2[0][0]), m * m, MPI_INT, 2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s3[0][0]), m * m, MPI_INT, 3, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s4[0][0]), m * m, MPI_INT, 4, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s5[0][0]), m * m, MPI_INT, 5, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s6[0][0]), m * m, MPI_INT, 6, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(s7[0][0]), m * m, MPI_INT, 7, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank == 1)
    {
        int** bds = SubMatrix(m, b, d);
        int** gha = SumMatrix(m, g, h);
        s1 = StrassenImp(m, bds, gha);
        DeleteMatrix(m, bds);
        DeleteMatrix(m, gha);
        MPI_Send(&(s1[0][0]), m * m, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }

    if (rank == 2)
    {
        int** ada = SumMatrix(m, a, d);
        int** eha = SumMatrix(m, e, h);
        s2 = StrassenImp(m, ada, eha);
        DeleteMatrix(m, ada);
        DeleteMatrix(m, eha);
        MPI_Send(&(s2[0][0]), m * m, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    if (rank == 3)
    {
        int** acs = SubMatrix(m, a, c);
        int** efa = SumMatrix(m, e, f);
        s3 = StrassenImp(m, acs, efa);
        DeleteMatrix(m, acs);
        DeleteMatrix(m, efa);
        MPI_Send(&(s3[0][0]), m * m, MPI_INT, 0, 3, MPI_COMM_WORLD);
    }

    if (rank == 4)
    {
        int** aba = SumMatrix(m, a, b);
        s4 = StrassenImp(m, aba, h);
        DeleteMatrix(m, aba);
        MPI_Send(&(s4[0][0]), m * m, MPI_INT, 0, 4, MPI_COMM_WORLD);
    }
    DeleteMatrix(m, b);

    if (rank == 5)
    {
        int** fhs = SubMatrix(m, f, h);
        s5 = StrassenImp(m, a, fhs);
        DeleteMatrix(m, fhs);
        MPI_Send(&(s5[0][0]), m * m, MPI_INT, 0, 5, MPI_COMM_WORLD);
    }
    DeleteMatrix(m, a);
    DeleteMatrix(m, f);
    DeleteMatrix(m, h);

    if (rank == 6)
    {
        int** ges = SubMatrix(m, g, e);
        s6 = StrassenImp(m, d, ges);
        DeleteMatrix(m, ges);
        MPI_Send(&(s6[0][0]), m * m, MPI_INT, 0, 6, MPI_COMM_WORLD);
    }
    DeleteMatrix(m, g);

    if (rank == 7)
    {
        int** cda = SumMatrix(m, c, d);
        s7 = StrassenImp(m, cda, e);
        DeleteMatrix(m, cda);
        MPI_Send(&(s7[0][0]), m * m, MPI_INT, 0, 7, MPI_COMM_WORLD);
    }
    DeleteMatrix(m, c);
    DeleteMatrix(m, d);
    DeleteMatrix(m, e);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        int** s1s2a = SumMatrix(m, s1, s2);
        int** s6s4s = SubMatrix(m, s6, s4);
        int** c11 = SumMatrix(m, s1s2a, s6s4s);
        DeleteMatrix(m, s1s2a);
        DeleteMatrix(m, s6s4s);

        int** c12 = SumMatrix(m, s4, s5);
        int** c21 = SumMatrix(m, s6, s7);
        int** s2s3s = SubMatrix(m, s2, s3);
        int** s5s7s = SubMatrix(m, s5, s7);
        int** c22 = SumMatrix(m, s2s3s, s5s7s);
        DeleteMatrix(m, s2s3s);
        DeleteMatrix(m, s5s7s);

        result = CombineMatrix(m, c11, c12, c21, c22);
        DeleteMatrix(m, c11);
        DeleteMatrix(m, c12);
        DeleteMatrix(m, c21);
        DeleteMatrix(m, c22);
    }

    DeleteMatrix(m, s1);
    DeleteMatrix(m, s2);
    DeleteMatrix(m, s3);
    DeleteMatrix(m, s4);
    DeleteMatrix(m, s5);
    DeleteMatrix(m, s6);
    DeleteMatrix(m, s7);
}

int** StrassenImp(int n, int** matrix1, int** matrix2)
{
    if (n <= 32)
    {
        return MulMatrix(n, matrix1, matrix2);
    }

    int m = n / 2;

    int** a = SliceMatrix(n, matrix1, 0, 0);
    int** b = SliceMatrix(n, matrix1, 0, m);
    int** c = SliceMatrix(n, matrix1, m, 0);
    int** d = SliceMatrix(n, matrix1, m, m);
    int** e = SliceMatrix(n, matrix2, 0, 0);
    int** f = SliceMatrix(n, matrix2, 0, m);
    int** g = SliceMatrix(n, matrix2, m, 0);
    int** h = SliceMatrix(n, matrix2, m, m);

    int** bds = SubMatrix(m, b, d);
    int** gha = SumMatrix(m, g, h);
    int** s1 = StrassenImp(m, bds, gha);
    DeleteMatrix(m, bds);
    DeleteMatrix(m, gha);

    int** ada = SumMatrix(m, a, d);
    int** eha = SumMatrix(m, e, h);
    int** s2 = StrassenImp(m, ada, eha);
    DeleteMatrix(m, ada);
    DeleteMatrix(m, eha);

    int** acs = SubMatrix(m, a, c);
    int** efa = SumMatrix(m, e, f);
    int** s3 = StrassenImp(m, acs, efa);
    DeleteMatrix(m, acs);
    DeleteMatrix(m, efa);

    int** aba = SumMatrix(m, a, b);
    int** s4 = StrassenImp(m, aba, h);
    DeleteMatrix(m, aba);
    DeleteMatrix(m, b);

    int** fhs = SubMatrix(m, f, h);
    int** s5 = StrassenImp(m, a, fhs);
    DeleteMatrix(m, fhs);
    DeleteMatrix(m, a);
    DeleteMatrix(m, f);
    DeleteMatrix(m, h);

    int** ges = SubMatrix(m, g, e);
    int** s6 = StrassenImp(m, d, ges);
    DeleteMatrix(m, ges);
    DeleteMatrix(m, g);

    int** cda = SumMatrix(m, c, d);
    int** s7 = StrassenImp(m, cda, e);
    DeleteMatrix(m, cda);
    DeleteMatrix(m, c);
    DeleteMatrix(m, d);
    DeleteMatrix(m, e);

    int** s1s2a = SumMatrix(m, s1, s2);
    int** s6s4s = SubMatrix(m, s6, s4);
    int** c11 = SumMatrix(m, s1s2a, s6s4s);
    DeleteMatrix(m, s1s2a);
    DeleteMatrix(m, s6s4s);
    DeleteMatrix(m, s1);

    int** c12 = SumMatrix(m, s4, s5);
    DeleteMatrix(m, s4);

    int** c21 = SumMatrix(m, s6, s7);
    DeleteMatrix(m, s6);

    int** s2s3s = SubMatrix(m, s2, s3);
    int** s5s7s = SubMatrix(m, s5, s7);
    int** c22 = SumMatrix(m, s2s3s, s5s7s);
    DeleteMatrix(m, s2s3s);
    DeleteMatrix(m, s5s7s);
    DeleteMatrix(m, s2);
    DeleteMatrix(m, s3);
    DeleteMatrix(m, s5);
    DeleteMatrix(m, s7);

    int** result = CombineMatrix(m, c11, c12, c21, c22);

    DeleteMatrix(m, c11);
    DeleteMatrix(m, c12);
    DeleteMatrix(m, c21);
    DeleteMatrix(m, c22);

    return result;
}

int** CreateMatrix(int n)
{
    int* data = (int*)malloc(n * n * sizeof(int));
    int** array = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++)
    {
        array[i] = &(data[n * i]);
    }
    return array;
}

void FillMatrix(int n, int**& matrix)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            matrix[i][j] = rand() % 5;
        }
    }
}

void DeleteMatrix(int n, int** matrix)
{
    free(matrix[0]);
    free(matrix);
}

int** SliceMatrix(int n, int** matrix, int offseti, int offsetj)
{
    int m = n / 2;
    int** slice = CreateMatrix(m);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            slice[i][j] = matrix[offseti + i][offsetj + j];
        }
    }
    return slice;
}

int** SumMatrix(int n, int** matrix1, int** matrix2)
{
    int** result = CreateMatrix(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }

    return result;
}

int** SubMatrix(int n, int** matrix1, int** matrix2)
{
    int** result = CreateMatrix(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }

    return result;
}

int** MulMatrix(int n, int** matrix1, int** matrix2)
{
    int** result = CreateMatrix(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result[i][j] = 0;
            for (int k = 0; k < n; k++)
            {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

int** CombineMatrix(int m, int** c11, int** c12, int** c21, int** c22)
{
    int n = 2 * m;
    int** result = CreateMatrix(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < m && j < m)
                result[i][j] = c11[i][j];
            else if (i < m)
                result[i][j] = c12[i][j - m];
            else if (j < m)
                result[i][j] = c21[i - m][j];
            else
                result[i][j] = c22[i - m][j - m];
        }
    }

    return result;
}

void PrintMatrix(int n, int** matrix)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}