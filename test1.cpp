#include "MapReduceClient.h"
#include "MapReduceFramework.h"
#include <stdlib.h>
#include <iostream>
#include <map>
#include <array>
#include <unistd.h>

#define N 10000
#define RANGE 200
#define THREADS 400

using namespace std;

struct Number : public K1, public K2, public K3, public V1, public V2, public V3 {

    int n;

    bool operator< (const K1 &other) const
    {
      return n < ((Number &) other).n;
    }

    bool operator< (const K2 &other) const
    {
      return n < ((Number &) other).n;
    }

    bool operator< (const K3 &other) const
    {
      return n < ((Number &) other).n;
    }

    ~Number ()
    {

    }
};

struct MRNumber : public MapReduceClient {
    void Map (const K1 *const key, const V1 *const val, void *context) const
    {
      auto k = new Number ();
      k->n = ((Number *) key)->n;
      auto v = new Number ();
      v->n = 1;
      usleep(150000);
      emit2 (k, v, context);
    }

    void Reduce (const K2 *key, const IntermediateVec *vals, void *context) const
    {
      auto k = (Number *) key;
      K3 *k3 = new Number ();
      V3 *v3 = new Number ();
      ((Number *) k3)->n = k->n;
      int count = 0;
      for (auto &val:*vals)
      {
        (void) val;
        count++;
        delete val.first;
        delete val.second;
      }
      ((Number *) v3)->n = count;
      emit3 (k3, v3, context);
    }
    virtual void map (const K1 *key, const V1 *value, void *context) const override
    {
      Map (key, value, context);
    }

    // gets a single K2 key and a vector of all its respective V2 values
    // calls emit3(K3, V3, context) any number of times (usually once)
    // to output (K3, V3) pairs.
    virtual void reduce (const IntermediateVec *pairs, void *context) const override
    {
      if (!pairs->empty ())
      {
        Reduce ((*pairs)[0].first, pairs, context);
      }
    }
};

int main ()
{
  InputVec numbers;
  MRNumber m;
  srand (0);
  for (int i = 0; i < N; ++i)
  {
    int n = std::rand () % RANGE + 1;
    auto numKey = new Number ();
    auto numVal = new Number ();
    numKey->n = n;
    numVal->n = 1;
    numbers.push_back (make_pair (numKey, numVal));
  }
  OutputVec results;
  auto job = startMapReduceJob (m, numbers, results, THREADS);
  waitForJob (job);
  closeJobHandle (job);
  std::map<int, int> expectedOutput = {
      {1, 60},
      {2, 45},
      {3, 57},
      {4, 49},
      {5, 52},
      {6, 57},
      {7, 49},
      {8, 46},
      {9, 44},
      {10, 44},
      {11, 50},
      {12, 48},
      {13, 50},
      {14, 46},
      {15, 46},
      {16, 50},
      {17, 51},
      {18, 53},
      {19, 61},
      {20, 43},
      {21, 47},
      {22, 57},
      {23, 43},
      {24, 55},
      {25, 40},
      {26, 49},
      {27, 59},
      {28, 55},
      {29, 66},
      {30, 53},
      {31, 48},
      {32, 47},
      {33, 50},
      {34, 53},
      {35, 44},
      {36, 52},
      {37, 52},
      {38, 46},
      {39, 54},
      {40, 46},
      {41, 53},
      {42, 35},
      {43, 52},
      {44, 43},
      {45, 48},
      {46, 52},
      {47, 53},
      {48, 57},
      {49, 35},
      {50, 48},
      {51, 51},
      {52, 55},
      {53, 47},
      {54, 52},
      {55, 47},
      {56, 52},
      {57, 46},
      {58, 69},
      {59, 47},
      {60, 51},
      {61, 67},
      {62, 46},
      {63, 40},
      {64, 35},
      {65, 51},
      {66, 47},
      {67, 42},
      {68, 55},
      {69, 42},
      {70, 64},
      {71, 43},
      {72, 51},
      {73, 65},
      {74, 50},
      {75, 39},
      {76, 48},
      {77, 54},
      {78, 41},
      {79, 47},
      {80, 43},
      {81, 49},
      {82, 62},
      {83, 50},
      {84, 52},
      {85, 50},
      {86, 52},
      {87, 59},
      {88, 51},
      {89, 50},
      {90, 41},
      {91, 57},
      {92, 50},
      {93, 63},
      {94, 47},
      {95, 59},
      {96, 42},
      {97, 48},
      {98, 50},
      {99, 53},
      {100, 46},
      {101, 49},
      {102, 47},
      {103, 51},
      {104, 47},
      {105, 62},
      {106, 41},
      {107, 41},
      {108, 49},
      {109, 58},
      {110, 42},
      {111, 49},
      {112, 43},
      {113, 43},
      {114, 39},
      {115, 41},
      {116, 54},
      {117, 48},
      {118, 43},
      {119, 50},
      {120, 46},
      {121, 54},
      {122, 39},
      {123, 48},
      {124, 62},
      {125, 55},
      {126, 54},
      {127, 45},
      {128, 58},
      {129, 62},
      {130, 51},
      {131, 50},
      {132, 48},
      {133, 51},
      {134, 43},
      {135, 49},
      {136, 45},
      {137, 51},
      {138, 49},
      {139, 63},
      {140, 63},
      {141, 55},
      {142, 51},
      {143, 55},
      {144, 47},
      {145, 40},
      {146, 38},
      {147, 59},
      {148, 42},
      {149, 38},
      {150, 53},
      {151, 51},
      {152, 58},
      {153, 46},
      {154, 42},
      {155, 67},
      {156, 45},
      {157, 40},
      {158, 54},
      {159, 52},
      {160, 53},
      {161, 53},
      {162, 47},
      {163, 45},
      {164, 50},
      {165, 54},
      {166, 46},
      {167, 53},
      {168, 58},
      {169, 55},
      {170, 44},
      {171, 52},
      {172, 47},
      {173, 49},
      {174, 48},
      {175, 56},
      {176, 43},
      {177, 55},
      {178, 56},
      {179, 28},
      {180, 48},
      {181, 57},
      {182, 46},
      {183, 50},
      {184, 46},
      {185, 55},
      {186, 42},
      {187, 66},
      {188, 51},
      {189, 39},
      {190, 63},
      {191, 58},
      {192, 58},
      {193, 47},
      {194, 53},
      {195, 54},
      {196, 39},
      {197, 65},
      {198, 58},
      {199, 43},
      {200, 43}
  };
  for (OutputPair &pair : results)
  {
    int c = ((Number *) pair.first)->n;
    int count = ((Number *) pair.second)->n ;
    std::cout  << c << ": " << count << std::endl;
    auto iter = expectedOutput.find(c);
    if (iter != expectedOutput.end())
    {
      // element found;
      if (count != iter->second)
      {
        std::cout << "ERROR OF KEY:" << c <<std::endl<< "ACTUAL VALUE: " << count << ", EXPECTED VALUE: " << iter->second << std::endl;
        exit(EXIT_FAILURE);
      }
      else
      {
          expectedOutput.erase(iter);
      }
    }
    else
    {
      std::cout << "ERROR: THE KEY " << c << " WITH VALUE " << count << " DOES NOT EXIST." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  if (!expectedOutput.empty())
  {
    std::cout << "ERROR: YOU MISS SOME KEYS IN YOUR OUTPUT VEC" << std::endl;
    exit(EXIT_FAILURE);

  }
  for (auto &i : results)
  {
    delete i.first;
    delete i.second;
  }

  for (int j = 0; j < N; ++j)
  {
    delete numbers[j].first;
    delete numbers[j].second;
  }
  std::cout << "PASSED THE TEST!" << std::endl;


  return 0;
}

