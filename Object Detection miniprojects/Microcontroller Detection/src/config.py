import torch

BATCH_SIZE = 10  # GPU Memory size
RESIZE_TO = 512  # resize the image training and transforms
NUM_EPOCHS = 100 # number of epochs to train for
NUM_WORKER = 2   # number of cpu workers

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train image and xml files directory
TRAIN_DIR = '..\\Microcontroller Detection\\train'

# Validation image and xml files directory
VALID_DIR = '..\\Microcontroller Detection\\test'

CLASSES = ['background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora']
NUM_CLASSES = 5
# two stage detection이라 배경도 봄

CARD_TRAIN_DIR = '..\\PokerCard Detection\\train'
CARD_VALID_DIR = '..\\PokerCard Detection\\valid'
CARD_TEST_DIR = '..\\PokerCard Detection\\test'

CARD_CLASSES = ['background','10 Diamonds', '10 Hearts', '10 Spades', '10 Trefoils','2 Diamonds', '2 Hearts', '2 Spades',
                '2 Trefoils','3 Diamonds', '3 Hearts', '3 Spades', '3 Trefoils',
                '4 Diamonds', '4 Hearts', '4 Spades', '4 Trefoils','5 Diamonds',
                '5 Hearts', '5 Spades', '5 Trefoils', '59','6 Diamonds', '6 Hearts', '6 Spades', '6 Trefoils',
                '7 Diamonds', '7 Hearts', '7 Spades', '7 Trefoils',
                '8 Diamonds', '8 Hearts', '8 Spades', '8 Trefoils',
                '9 Diamonds', '9 Hearts', '9 Spades', '9 Trefoils',
                'A Diamonds', 'A Hearts', 'A Spades', 'A Trefoils',
                'J Diamonds', 'J Hearts', 'J Spades', 'J Trefoils',
                'K Diamonds', 'K Hearts', 'K Spades', 'K Trefoils',
                'Q Diamonds', 'Q Hearts', 'Q Spades', 'Q Trefoils']
CARD_NUM_CLASSES = 53

# 데이터 로더 생성 후 이미지 시각화 여부
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = '..\\outputs'
SAVE_PLOTS_EPOCH = 2
SAVE_MODEL_EPOCH = 2
# save loss plots, models after these many epochs

NUM_SAMPLES_TO_VISUALIZE = 5