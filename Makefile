CC = clang

SRC_DIR = ${PWD}/src
INCLUDE_DIR = ${PWD}/include
BUILD_DIR = ${PWD}/build
LIB_DIR = ${PWD}/lib

CFLAGS += -O2 -g -Wall -Wextra -Wpedantic -std=c11 -march=native
LDFLAGS += -lgsl -lgslcblas -lm

SOURCES = ${SRC_DIR}/nn_math.c \
          ${SRC_DIR}/nn_network.c \
          ${SRC_DIR}/nn_utils.c

TEST_DIR = ${PWD}/tests

TEST_CFLAGS = -I${INCLUDE_DIR} -Werror -Wshadow
TEST_LDFLAGS = -L${LIB_DIR} ${LDFLAGS} -lcmocka -lnn

TEST_SOURCES = ${TEST_DIR}/test_helpers.c     \
               ${TEST_DIR}/nn_math_tests.c    \
               ${TEST_DIR}/nn_network_tests.c \
               ${TEST_DIR}/main.c

.SUFFIXES:
.SUFFIXES: .c .o

OBJECTS = ${SOURCES:.c=.o}

TEST_OBJECTS = ${TEST_SOURCES:.c=.o}

all: libnn.a

.c.o:
	${CC} -c ${CFLAGS} -I${INCLUDE_DIR} $< -o $@

libnn.a: ${OBJECTS}
	mkdir -p ${LIB_DIR}
	${AR} -cq ${LD_FLAGS} ${LIB_DIR}/$@ $^

tests: clean libnn.a
	mkdir -p ${BUILD_DIR}
	${CC} ${TEST_SOURCES} ${TEST_CFLAGS} ${TEST_LDFLAGS} -o ${BUILD_DIR}/$@
	${BUILD_DIR}/$@

mem_tests: tests
	valgrind                \
	  --error-exitcode=1    \
	  --leak-check=full     \
	  --show-leak-kinds=all \
	  ${BUILD_DIR}/tests

.PHONY: clean
clean:
	rm -f ${SRC_DIR}/*.o
	rm -f ${LIB_DIR}/libnn.a
	rm -f ${BUILD_DIR}/tests

.PHONY: distclean
distclean: clean
	rmdir ${LIB_DIR}
	rmdir ${BUILD_DIR}
