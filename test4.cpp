/**
 * Map-reduce Word-Frequencies
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include "MapReduceClient.h"
#include "MapReduceFramework.h"
#define NUM_THREADS 10

class Line : public K1 {
 private:
  const std::string line;
 public:
  Line (const std::string &line) : line (line)
  {}

  virtual bool operator< (const K1 &other) const
  {
    return this->line < ((Line &) other).line;
  }

  const std::string &getLine ()
  {
    return line;
  }
};

class Word : public K2, public K3 {
 private:
  const std::string word;
 public:
  Word (const std::string &word) : word (word)
  {}

  Word (const Word &other) = default;

  virtual bool operator< (const K2 &other) const
  {
    return this->word < ((Word &) other).word;
  }

  virtual bool operator< (const K3 &other) const
  {
    return this->word < ((Word &) other).word;
  }

  const std::string &getWord ()
  {
    return word;
  }

};

class Integer : public V3 {
 public:
  int val;

  Integer (int val) : val (val)
  {}
};

class MapReduceWordFrequencies : public MapReduceClient {
  virtual void Map (const K1 *key, const V1 *val, void *context) const
  {
    std::stringstream sstream (((Line *) key)->getLine ());
    std::string word;
    while (sstream >> word)
    {
      Word *k2 = new Word (word);
      emit2 (k2, nullptr, context);
    }
  }

  virtual void Reduce (const K2 *const key, const IntermediateVec *vals, void *context) const
  {
    Word *k3 = new Word (((Word &) *key));
    auto *frequency = new Integer ((*vals).size ());
    for (auto &val:*vals)
    {
      delete val.first;
      delete val.second;
    }
    emit3 (k3, frequency, context);
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

void writeByFrequency (OutputVec &frequencies, std::ofstream &ofs)
{
  // get length of longest word (so we can write to the file in a nice format)
  unsigned int maxLength = 0;
  for (auto it = frequencies.begin (); it != frequencies.end (); ++it)
  {
    unsigned int length = (*(Word *) it->first).getWord ().length ();
    maxLength = length > maxLength ? length : maxLength;
  }

  // sort by frequency is descending order
  std::sort (
      frequencies.begin (),
      frequencies.end (),
      [] (const OutputPair &o1, const OutputPair &o2)
      {

        if (((Integer *) o1.second)->val < ((Integer *) o2.second)->val)
          return false;

        return ((Integer *) o1.second)->val > ((Integer *) o2.second)->val
        || ((Word *) o1.first)->getWord () < ((Word *) o2.first)->getWord ();
      }
      );

  // writing results to file
  for (auto it = frequencies.begin (); it != frequencies.end (); ++it)
  {
    const std::string &word = (*(Word *) it->first).getWord ();
    int frequency = ((Integer *) it->second)->val;
    ofs << '{' << word  << " , " << frequency << "}" << std::endl;
  }
}

void findWordFrequencies (std::ifstream &ifs, std::ofstream &ofs)
{
  InputVec k1v1Pairs;

  // make the input for the framework
  std::string line;
  while (std::getline (ifs, line))
  {
    Line *k1 = new Line (line);
    k1v1Pairs.push_back (InputPair (k1, nullptr));
  }

  MapReduceWordFrequencies mapReduceObj;

  OutputVec frequencies;
  auto job = startMapReduceJob (mapReduceObj, k1v1Pairs, frequencies, NUM_THREADS);

  waitForJob (job);
  closeJobHandle (job);
  writeByFrequency (frequencies, ofs);
  for(auto pair: k1v1Pairs){
      delete pair.first;
    }
    for(auto pair: frequencies){
      delete pair.first;
      delete pair.second;
    }
}

bool compareFiles(const std::string& file1Path, const std::string& file2Path) {
    std::ifstream file1(file1Path);
    std::ifstream file2(file2Path);
    int i = 0;

    if (!file1.is_open() || !file2.is_open()) {
        std::cout << "Error opening files!" << std::endl;
        return false;
    }

    std::string line1, line2;
    while (std::getline(file1, line1) && std::getline(file2, line2)) {
        i++;
        if (line1 != line2) {
            return false;
        }
    }

    // Check if one file reached the end while the other still has more content
    if (file1.eof() != file2.eof()) {
        return false;
    }

    return true;
}

int main (int argc, char *argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: WordFrequencies <path to text_file>" << std::endl;
    return 1;
  }

  std::string textPath (argv[1]);

  std::ifstream ifs (textPath);
  std::ofstream ofs (textPath + std::string ("_test_results"),
                     std::ofstream::out);

  if (!ifs.is_open () || !ofs.is_open ())
  {
    std::cerr << "ERROR: can't open input or output file"
    << std::endl;
    return 1;
  }

  findWordFrequencies (ifs, ofs);
  ifs.close ();
  ofs.close ();


  std::string file1Path = textPath + std::string ("_test_results");
  std::string file2Path = argv[2];
  bool isEqual = compareFiles(file1Path, file2Path);
  if (isEqual) {
      std::cout << "The files have the same content." << std::endl;
  } else {
      std::cout << "The files have different content." << std::endl;
  }

}
