#include <iostream>
#include <vector>
#include <time.h>
using namespace std;


vector<double> gauss(vector<vector<double>>& a, vector<double>& b) {
    int length = a.size();//кол-во неизвестных
    vector<double> result(length);//тут будет результат

    //Прямой ход
    for (int k = 0; k < length - 1; k++) {
        double del = a[k][k];
        for (int i = k + 1; i < length; i++) {
            double mult = a[i][k] / del;
            for (int j = k; j < length; j++) {
                a[i][j] -= mult * a[k][j];
            }
            b[i] -= mult * b[k];
        }
    }

    //Обратный ход
    for (int k = length - 1; k >= 0; k--) {
        result[k] = b[k];
        for (int i = k + 1; i < length; i++) {
            result[k] -= a[k][i] * result[i];
        }
        result[k] /= a[k][k];
    }

    return result;
}

vector<double> gauss_parallel(vector<vector<double>>& a, vector<double>& b) {
    int length = a.size();//кол-во неизвестных
    vector<double> result(length);//тут будет результат
    double temp = 0;

    //Прямой ход
    #pragma omp parallel
    for (int k = 0; k < length - 1; k++) {
        double del = a[k][k];
        for (int i = k + 1; i < length; i++) {
            double mult = a[i][k] / del;
            #pragma omp for
            for (int j = k; j < length; j++) {
                a[i][j] -= mult * a[k][j];
            }
            b[i] -= mult * b[k];
        }
    }

    //Обратный ход
    #pragma omp parallel
    for (int k = length - 1; k >= 0; k--) {
        temp = 0;
        #pragma omp barrier 

        #pragma omp for reduction(+:temp)
        for (int i = k + 1; i < length; i++) {
            temp += a[k][i] * result[i];
        }
        #pragma omp single
        result[k] = (b[k] - temp) / a[k][k];
    }

    return result;
}
    

int main() {
    time_t start, end;
    int n = 1500;
    vector<vector<double>> a(n, vector<double>(n));
    vector<double> b(n);
    for (int i = 0; i < n; i++) {
        //srand(1);
        for (int j = 0; j < n; j++) {
            a[i][j] = rand() % 100 + 1;
        }
        b[i] = rand() % 100 + 1;
    }
    time(&start);
    vector<double> res = gauss(a, b);
    time(&end);
    //for (int i = 0; i < n; i++) {
    //    cout << res[i] << " ";
    //}
    cout << "Time(guass): " << difftime(end, start) << " seconds";
    cout << endl;
    time(&start);
    res = gauss_parallel(a, b);
    time(&end);
    //for (int i = 0; i < n; i++) {
    //    cout << res[i] << " ";
    //}
    cout << "Time(guass_parallel): " << difftime(end, start) << " seconds";
    cout << endl;
    return 0;
}