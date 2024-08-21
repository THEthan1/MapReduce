//
// Created by Ethan Glick & Yaakov Kosman on 11/05/2023.
//

#include "MapReduceFramework.h"
#include <pthread.h>
#include <vector>
#include <atomic>
#include <algorithm>
#include <queue>
#include <iostream>
#include <map>
#include <list>

#define GET_RIGHTMOST_31_BITS(x) ((unsigned int) ((x << 33) >> 33))
#define GET_MIDDLE_31_BITS(x) ((unsigned int) ((x << 2)>> 33))
#define GET_STAGE_BITS(x) ((unsigned int) (x >> 62))
#define SET_ELEMENTS(a,b,c) ((uint64_t) (((uint64_t) a << 62) | ((uint64_t) b << 31) | (uint64_t) c))

typedef struct JobContext JobContext;
typedef struct {
    IntermediateVec * intermediateVec;
    int tid;
    JobContext *jobContext;
} ThreadContext;


struct JobContext{
    pthread_t* threads;
    ThreadContext* threadContexts;
    pthread_cond_t* map_to_shuffle_non_base_barrier;
    pthread_cond_t* map_to_shuffle_base_barrier;
    pthread_mutex_t* map_to_shuffle_general_mutex;
    pthread_mutex_t* reduce_mutex;
    pthread_mutex_t* emit3_mutex;
    unsigned int* progress_counter;
    const int num_of_threads;
    std::vector<IntermediateVec*> *shuffled_vector_list;
    // READ RIGHT TO LEFT:
    // 2-bits = stage, 31-bits = remaining, 31-bits = processed <-- READ THIS WAY
    std::atomic<uint64_t>* atomic_counter;
    const MapReduceClient* client;
    const InputVec * inputVec;
    OutputVec * outputVec;
};

void emit2 (K2* key, V2* value, void* context) {
    auto * threadContext = (ThreadContext *)context;
    threadContext->intermediateVec->push_back(IntermediatePair(key, value));
}

void emit3 (K3* key, V3* value, void* context) {
    auto * threadContext = (ThreadContext *)context;

    if (pthread_mutex_lock(threadContext->jobContext->emit3_mutex)) {
        std::cout<<"system error: failed to lock mutex\n";
        exit(1);
    }

    //only one thread can do this at a time
    threadContext->jobContext->outputVec->push_back(OutputPair(key, value));

    if (pthread_mutex_unlock(threadContext->jobContext->emit3_mutex)) {
        std::cout<<"system error: failed to unlock mutex\n";
        exit(1);
    }
}

unsigned int get_total_number_of_intermediate_pairs(JobContext* jobContext){
    unsigned int total_number = 0;
    for (int i = 0; i < jobContext->num_of_threads; i++){
        total_number += jobContext->threadContexts[i].intermediateVec->size();
    }
    return total_number;
}

/**
 * this function returns the next index of a value in the input vector that was not mapped yet.
 * although the function itself is NOT atomic, it DOES ensure that if it is return a value,
 * no other thread will ever use that value and any other value smaller
 * than it. it modifies the atomic variable exactly once.
 *
 * @param jobContext the entire job context
 * @return an index in the inputVec that haven't been mapped yet,-1 if all values have been mapped.
 */
long getNextAtomicValue(JobContext *jobContext){
    uint64_t current_atomic_value  = jobContext->atomic_counter->load();
    unsigned int target_value = GET_MIDDLE_31_BITS(current_atomic_value);
    unsigned int current_count = GET_RIGHTMOST_31_BITS(current_atomic_value);
    while (target_value > current_count){//so when we took the atomic value the entire
        // inputVe haven't been mapped fully
        if(jobContext->atomic_counter->compare_exchange_weak(current_atomic_value, current_atomic_value + 1)){
            //the condition above will return true if and only if the value of the atomic
            // variable was not changed since we took
            // last its value. And in that case, the atomic variable value is incremented -
            // so no other thread will ever get that value again
            return current_count;
        }else{//some other thread already incremented the atomic counter and took that value
            current_atomic_value = jobContext->atomic_counter->load();//try to get the next available value
            current_count = GET_RIGHTMOST_31_BITS(current_atomic_value);
        }
    }

    return -1;//all values have been allocated
}

void runMap(ThreadContext *threadContext) {
    const InputVec *inputVec = threadContext->jobContext->inputVec;
    long value = getNextAtomicValue(threadContext->jobContext);

    while(value != -1) {
        threadContext->jobContext->client->map(inputVec->at(value).first,
                                               inputVec->at(value).second,
                                               (void *)threadContext);
        value = getNextAtomicValue(threadContext->jobContext);
    }
}

void runShuffle(ThreadContext *threadContext) {
    int num_threads = threadContext->jobContext->num_of_threads;
    auto atomic_counter = threadContext->jobContext->atomic_counter;
    atomic_counter->store(SET_ELEMENTS(
                                  SHUFFLE_STAGE, get_total_number_of_intermediate_pairs(threadContext->jobContext), 0));

    std::vector<IntermediatePair>* vectors[num_threads];
    for (int i = 0; i < num_threads; i++) {
        vectors[i] = threadContext->jobContext->threadContexts[i].intermediateVec;
    }

    int cur_index = 0;
    K2* prev_max = nullptr;
    auto shuffled_vec = new std::vector<IntermediatePair>();
    while (true) {
        K2* max = nullptr;
        int empty_count = 0;
        for (int i = 0; i < num_threads; i++) {
            if (vectors[i]->empty()) empty_count++;
            else if (max == nullptr || *max < *vectors[i]->back().first) {
                max = vectors[i]->back().first;
                cur_index = i;
            }
        }

        if (max && prev_max && *max < *prev_max) {
            threadContext->jobContext->shuffled_vector_list->push_back(shuffled_vec);
            shuffled_vec = new std::vector<IntermediatePair>();
        }
        else if (empty_count == num_threads) {
            threadContext->jobContext->shuffled_vector_list->push_back(shuffled_vec);
            return;
        }

        prev_max = max;

        auto key = vectors[cur_index]->back().first;

        for (int j = 0; j < num_threads; j++) {
            if (vectors[j]->empty()) continue;

            // if keys match
            if (!(*vectors[j]->back().first < *key) && !(*key < *vectors[j]->back().first)) {
                shuffled_vec->push_back(vectors[j]->back());
                vectors[j]->pop_back();

                (*atomic_counter)++;
            }
        }
    }
}

void runReduce(ThreadContext *threadContext) {
    unsigned int total_pairs = GET_MIDDLE_31_BITS(threadContext->jobContext->atomic_counter->load());
    unsigned long complete_shuffle_stage_pattern = SET_ELEMENTS(SHUFFLE_STAGE, total_pairs, total_pairs);
    unsigned long empty_reduce_stage_pattern = SET_ELEMENTS(REDUCE_STAGE, total_pairs, 0);

    auto atomic_counter = threadContext->jobContext->atomic_counter;
    atomic_counter->compare_exchange_weak(complete_shuffle_stage_pattern, empty_reduce_stage_pattern);

    auto shuffled_vector_list = threadContext->jobContext->shuffled_vector_list;
    IntermediateVec* intermediateVec;

    while(true) {
        if (pthread_mutex_lock(threadContext->jobContext->reduce_mutex)) {
            std::cout<<"system error: failed to lock mutex\n";
            exit(1);
        }

        if ((size_t)(*threadContext->jobContext->progress_counter) == shuffled_vector_list->size()) {
            if (pthread_mutex_unlock(threadContext->jobContext->reduce_mutex)) {
                std::cout<<"system error: failed to unlock mutex\n";
                exit(1);
            }

            return;
        }

        intermediateVec = shuffled_vector_list->at((*threadContext->jobContext->progress_counter)++);

        if (pthread_mutex_unlock(threadContext->jobContext->reduce_mutex)) {
            std::cout<<"system error: failed to unlock mutex\n";
            exit(1);
        }

        unsigned int size = intermediateVec->size();
        threadContext->jobContext->client->reduce(intermediateVec, (void *) threadContext);
        (*atomic_counter) += size;
    }
}

void mapToReduceBarrier(ThreadContext *threadContext) {
    if (pthread_mutex_lock(threadContext->jobContext->map_to_shuffle_general_mutex)) { //lock the mutex
        std::cout<<"system error: failed to lock mutex\n";
        exit(1);
    }

    (*threadContext->jobContext->progress_counter)++;//increase the number of threads that reached this point
    if (*threadContext->jobContext->progress_counter == (unsigned int) threadContext->jobContext->num_of_threads){
        //so all threads including the base thread have reached this point and have ended their sort phase.
        //release the base thread from its barrier if it is locked in it
        // (it is not locked in it if the base thread have reached the barrier last)
        if (pthread_cond_signal(threadContext->jobContext->map_to_shuffle_base_barrier)) {
            std::cout<<"system error: failed to send conditional signal\n";
            exit(1);
        }

        if(threadContext->tid != 0){
            //so a non-base thread have reached the barrier last and released the base thread,
            // then it must be blocked until the base thread will end its shuffle phase.
            if (pthread_cond_wait(threadContext->jobContext->map_to_shuffle_non_base_barrier,
                                  threadContext->jobContext->map_to_shuffle_general_mutex)) {
                std::cout<<"system error: failed to initiate conditional wait\n";
                exit(1);
            }
        }
    }
    else{
        //if not all the threads have ended their sorting phase
        if(threadContext->tid == 0){
            //block the base thread with a unique barrier
            if (pthread_cond_wait(threadContext->jobContext->map_to_shuffle_base_barrier,
                                  threadContext->jobContext->map_to_shuffle_general_mutex)) {
                std::cout<<"system error: failed to initiate conditional wait\n";
                exit(1);
            }
        }
        else {
            //block the current non-base thread until the main thread will complete the
            // shuffle stage and awake all the blocked threads together
            if (pthread_cond_wait(threadContext->jobContext->map_to_shuffle_non_base_barrier,
                                  threadContext->jobContext->map_to_shuffle_general_mutex)) {
                std::cout<<"system error: failed to initiate conditional wait\n";
                exit(1);
            }
        }
    }

    if (pthread_mutex_unlock(threadContext->jobContext->map_to_shuffle_general_mutex)) { //unlock the mutex
        std::cout<<"system error: failed to unlock mutex\n";
        exit(1);
    }
}

bool pairCompare(const std::pair<K2*,V2*> &a, const std::pair<K2*,V2*> &b)
{
    return (*(a.first) < *(b.first));
}

void * runMultiThreadedMapReduce(void * context){
    auto* threadContext = (ThreadContext *)context;
    // (const uint64_t) UNDEFINED_STAGE, (const uint64_t) inputVec.size(), 0};
    unsigned long undefined_stage_pattern = SET_ELEMENTS(UNDEFINED_STAGE,
                                                         threadContext->jobContext->inputVec->size(),0);
    unsigned long empty_map_stage_pattern = SET_ELEMENTS(MAP_STAGE, threadContext->jobContext->inputVec->size(),0);
    auto atomic_counter = threadContext->jobContext->atomic_counter;
    atomic_counter->compare_exchange_weak(undefined_stage_pattern, empty_map_stage_pattern);

    runMap(threadContext); // map stage
    // sort stage, thread safe.
    std::sort(threadContext->intermediateVec->begin(), threadContext->intermediateVec->end(), pairCompare);
    // BARRIER START-------------------------------------------------
    mapToReduceBarrier(threadContext);

    if (threadContext->tid == 0) {
        runShuffle(threadContext); // shuffle stage
        //release all the threads into the reduce stage after the base thread have finished processing
        // all the intermediate values
        *threadContext->jobContext->progress_counter = 0;
        pthread_cond_broadcast(threadContext->jobContext->map_to_shuffle_non_base_barrier);
    }

    // BARRIER END-------------------------------------------------
    runReduce(threadContext); // reduce stage

    return context;
}

JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel) {
    auto atomic_counter = new std::atomic<uint64_t>(0);
    atomic_counter->store(SET_ELEMENTS(UNDEFINED_STAGE, inputVec.size(), 0));
    auto *map_to_shuffle_barrier = new pthread_cond_t;
    auto *map_to_shuffle_zero_barrier = new pthread_cond_t;
    auto *map_to_shuffle_mutex =  new pthread_mutex_t;
    auto *reduce_mutex = new pthread_mutex_t;
    auto *emmit3_mutex = new pthread_mutex_t;

    if (pthread_cond_init(map_to_shuffle_barrier, nullptr) ||
        pthread_cond_init(map_to_shuffle_zero_barrier, nullptr) ||
        pthread_mutex_init(map_to_shuffle_mutex, nullptr) ||
        pthread_mutex_init(reduce_mutex, nullptr) ||
        pthread_mutex_init(emmit3_mutex, nullptr)) {
        std::cout<<"system error: failed to initialize mutex\n";
        exit(1);
    }

    auto job = new JobContext{new pthread_t[multiThreadLevel],
                              new ThreadContext[multiThreadLevel],
                              map_to_shuffle_barrier,
                              map_to_shuffle_zero_barrier,
                              map_to_shuffle_mutex,
                              reduce_mutex, emmit3_mutex,
                              new unsigned int(0), multiThreadLevel,
                              new std::vector<IntermediateVec*>(),
                              atomic_counter, &client, &inputVec, &outputVec};

    for (int i = 0; i < multiThreadLevel; i++) {
        job->threadContexts[i] = ThreadContext {new IntermediateVec,  i, job};
        if (pthread_create(&job->threads[i], nullptr,
                           runMultiThreadedMapReduce, &job->threadContexts[i])) {
            std::cout<<"system error: failed to create thread\n";
            exit(1);
        }
    }

    return (JobHandle *)job;
}

inline bool isDone(JobContext *jobContext) {
    uint64_t value = jobContext->atomic_counter->load();
    return (GET_STAGE_BITS(value) == REDUCE_STAGE) &&
           GET_RIGHTMOST_31_BITS(value) == GET_MIDDLE_31_BITS(value);
}

void waitForJob(JobHandle job) {
    auto jobContext = (JobContext *)job;
    if (!isDone(jobContext)) {
        for(int i = 0; i < jobContext->num_of_threads; i++){
            auto thread = jobContext->threads[i];
            jobContext->threads[i] = 0;
            if (pthread_join(thread, nullptr)) {
                std::cout<<"system error: failed to join thread\n";
                exit(1);
            }
        }
    }
    else {
        for(int i = 0; i < jobContext->num_of_threads; i++){
            if (jobContext->threads[i] == 0) {
                continue;
            }
            else if (pthread_join(jobContext->threads[i], nullptr)) {
                std::cout<<"system error: failed to join thread\n";
                exit(1);
            }
        }
    }
}

void getJobState(JobHandle job, JobState* state) {
    auto jobContext = (JobContext*)job;
    uint64_t value = jobContext->atomic_counter->load();
    state->stage = (stage_t) GET_STAGE_BITS(value);
    state->percentage = (float) ((double) GET_RIGHTMOST_31_BITS(value) /
                                 (double) GET_MIDDLE_31_BITS(value)) * 100.0f;
}

void closeJobHandle(JobHandle job) {
    waitForJob(job);
    auto jobContext = (JobContext*)job;
    auto threadContexts = jobContext->threadContexts;
    for (int i = 0; i < jobContext->num_of_threads; i++) {
        for (auto iter : *threadContexts[i].intermediateVec) {
            delete &iter;
        }

        delete threadContexts[i].intermediateVec;
    }

    for (auto iter : *jobContext->shuffled_vector_list) {
        delete iter;
    }

    if (pthread_mutex_destroy(jobContext->emit3_mutex) || pthread_mutex_destroy(jobContext->reduce_mutex)
        || pthread_mutex_destroy(jobContext->map_to_shuffle_general_mutex)
        || pthread_cond_destroy(jobContext->map_to_shuffle_base_barrier)
        || pthread_cond_destroy(jobContext->map_to_shuffle_non_base_barrier)) {
        std::cout<<"system error: failed to destroy mutex\n";
        exit(1);
    }

    delete[] threadContexts;
    delete[] jobContext->threads;
    delete jobContext->shuffled_vector_list;
    delete jobContext->atomic_counter;
    delete jobContext->progress_counter;
    delete jobContext->emit3_mutex;
    delete jobContext->reduce_mutex;
    delete jobContext->map_to_shuffle_general_mutex;
    delete jobContext->map_to_shuffle_base_barrier;
    delete jobContext->map_to_shuffle_non_base_barrier;
    delete jobContext;
}

