CC := g++
CFLAGS := -pthread -std=c++11
SRC := MapReduceFramework.cpp
OBJ := $(SRC:.cpp=.o)
LIB := libMapReduceFramework.a

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJ)
		ar rcs $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ): MapReduceFramework.h MapReduceClient.h

clean:
	rm -f $(OBJ) $(LIB)



