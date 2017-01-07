#include<stdio.h>
#include<string.h>
#include<math.h>

#include<vector>
using namespace std;

// Suppose, there is a string; s = {abcbe abcac abcde abbcd acbde}
// Then the occurence vector of 'b' is {2, 4, 7, 12, 14, 17, 18, 23}
// We will try to find the periodicity of pattern 'b'

// Suppose, the confidence threshold is 60%
// It means the pattern 'b' has to exist at least 3 times in 5 segment of String s

#define SIZE 8
double threshold = 0.6;
double perfectPeriodicity=5;
vector <int> periodValues;

int getPeriodValue()
{
    int occ_vec[SIZE] = {2, 4, 7, 12, 14, 17, 18, 23};
    int st;
    int i,j, m;
    int period =0;
    double frequencyCount =0;

    // FPPM ALGORITHM
    for(i=1; i<=SIZE-1; i++){
        st = occ_vec[i-1];          // starting position
        for(j=i; j<=SIZE-1; j++){
            int delta = occ_vec[j];
            period = delta - st;
            frequencyCount =0;

            for(m=j; m<= SIZE-1; m++){
                if( (occ_vec[m]-st)% period==0){
                frequencyCount++;
                }
            }
            // checking periodicity
                double conf = frequencyCount/ perfectPeriodicity;
              //  printf("%lf conf\n", conf);
                if( conf >= threshold ){
                        periodValues.push_back(period);
                        printf("Period %d, Confidence: %.2lf Support: %.2lf\n", period, conf, frequencyCount);
                }
        }
    }

return 0;
}

int main()
{
    getPeriodValue();
    puts("\nPeriod:");
    for(int i=0; i< periodValues.size() ; i++ )
			printf("\t %d\n", periodValues[i] );

return 0;
}
/*
In real life problem, we can generate suffix trie of a database
to find occurrence vector of each events
*/
