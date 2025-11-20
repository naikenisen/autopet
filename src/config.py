
PET_IMAGE_PATH =  "/work/imvia/in156281/data_unet/pet_images"
CT_IMAGE_PATH =  "/work/imvia/in156281/data_unet/ct_images"
LABEL_PATH =  "/work/imvia/in156281/data_unet/labels"
MODEL_SAVE_PATH = "/work/imvia/in156281/autopet/models/unet_model.pth"
SLICE_AXIS = 2  # 0: x, 1: y, 2: z
BATCH_SIZE = 20
NUM_WORKERS = 8
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.3
NUM_EPOCHS = 20
RANDOM_SEED = 42