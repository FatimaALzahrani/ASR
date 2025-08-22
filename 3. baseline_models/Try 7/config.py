import os
import pandas as pd
import numpy as np
import librosa
import json
import argparse
from collections import Counter, defaultdict
import warnings
from datetime import datetime
import pickle
import re
from scipy import signal
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

warnings.filterwarnings('ignore')

class Config:
    SAMPLE_RATE = 22050
    DURATION = 3.0
    RANDOM_STATE = 42
    
    SPEAKER_MAPPING = {
        'أحمد': list(range(0, 7)),
        'عاصم': list(range(7, 14)),
        'هيفاء': list(range(14, 21)),
        'أسيل': list(range(21, 29)),
        'وسام': list(range(29, 37))
    }
    
    AUDIO_EXTENSIONS = ('.wav', '.mp3', '.m4a', '.flac')
    MIN_SAMPLES_PER_WORD = 3
    TEST_SIZE = 0.3
    QUALITY_THRESHOLD = 0.2
    
    N_MFCC = 13
    N_MEL = 10
    N_CHROMA = 12
    N_SPECTRAL_CONTRAST = 7
    
    FEATURE_SELECTION_K = 30
    GLOBAL_FEATURE_SELECTION_K = 50