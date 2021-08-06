CC = g++
CFLAGS = -std=c++14 -g -Wall
ROOT_PATH = C:\\ProjectStudy\\mapper
MAIN_REPO_PATH_LIBS = C:\\ProjectStudy\\libraries
PRE_COMP	=	$(ROOT_PATH)\\precomp.h
MAPPER_IMAGE_LIB	=	$(ROOT_PATH)/image_lib
ROOT_PATH_INCLUDES	=	$(wildcard $(ROOT_PATH)/*.h)

LOCAL_INCLUDES =\
-I$(ROOT_PATH) \
-I$(MAPPER_IMAGE_LIB)

IMAGE_LIB_SRC = $(wildcard $(MAPPER_IMAGE_LIB)/*.h) $(wildcard $(MAPPER_IMAGE_LIB)/*.cpp)



#$(PRE_COMP)
SRCS	=	\
$(ROOT_PATH_INCLUDES)	\
$(IMAGE_LIB_SRC)

APP_SRCS	=	main.cpp	\
$(SRCS)

TEST_SRCS	=	\
$(SRCS)

OPENCV =\
-I$(MAIN_REPO_PATH_LIBS)/opencv40/install/include	\
-L$(MAIN_REPO_PATH_LIBS)/opencv40/lib	\
-lopencv_core400	\
-lopencv_highgui400	\
-lopencv_imgcodecs400	\
-lopencv_imgproc400	\
-lopencv_objdetect400	\
-lopencv_videoio400	\
-lopencv_video400	\
-lopencv_features2d400	\
# -lopencv_calib3d400	\
# -lopencv_gapi400	\


PROG = mapper
TEST = test
PCH_OUT = precomp.h.gch
OUTFOLDER= out
LIBS = $(LOCAL_INCLUDES) $(OPENCV) -lwsock32

# $(PCH_OUT):$(PRE_COMP)
# 	$(CC)	$(CFLAGS) -H -x $(PRE_COMP) $(OPENCV)

$(PROG):$(APP_SRCS)	$(PCH_OUT)
	$(CC)	$(CFLAGS)	$(APP_SRCS)	-o	$(OUTFOLDER)/$(PROG)	$(LIBS)

$(TEST):$(TEST_SRCS)
	$(CC)	$(CFLAGS)	$(TEST_SRCS)	-o	$(OUTFOLDER)/$(TEST)	$(LIBS)