#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Criado: 22/11/2019
Autor: Marcello Sandi Pinheiro
Ref.: 
- https://www.analyticsvidhya.com/blog/2019/07/learn-build-first-speech-to-text-model-python/
- https://www.analyticsvidhya.com/blog/2021/01/introduction-to-automatic-speech-recognition-and-natural-language-processing/
"""
from __future__ import print_function
import datetime
from matplotlib import cm, pyplot as plt

import os
import argparse
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import numpy as np
from scipy.io import wavfile

from hmmlearn import hmm
from python_speech_features import mfcc

# Lista contendo os scores de cada ciclo de teste
list_scores = []

# =============================================================================
# # 1) Função para ler os arquivos de entrada
# =============================================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Treina o sistema de reconhecimento de fala baseado em HMM')
    parser.add_argument("--input-folder", dest="input_folder", required=True,
            help="Pasta de entrada contendo os arquivos de áudio para treinamento")
    return parser

# =============================================================================
# # 2) Define a classe de trainamento do HMM
# =============================================================================
class ModelHMM(object):
    def __init__(self, num_components=4, num_iter=1000):
        self.n_components = num_components
        self.n_iter = num_iter

        self.cov_type = 'diag'
        self.model_name = 'GaussianHMM'

        self.models = []

        self.model = hmm.GaussianHMM(n_components=self.n_components,
                covariance_type=self.cov_type, n_iter=self.n_iter)

# =============================================================================
#     # 3) 'training_data' é uma matriz numpy 2D em que cada linha é 13D
# Incluído: Heurísticas de aproximação e maximização das probabilidades.
#           algorith=viterbi ou baum-welch (EM)
# =============================================================================
    #def train(self, training_data, algorithm='baum-welch'): # algorith=baum-welch
    def train(self, training_data, algorithm='viterbi'): # algorith=viterbi
        np.seterr(all='ignore')
        cur_model = self.model.fit(training_data)
        self.models.append(cur_model)

    # 4) Executa o modelo HMM para inferência dos dados de entrada
    def compute_score(self, input_data):
        return self.model.score(input_data)
              
    def compute_predict(self, input_data):
        return self.model.predict(input_data)
    
    def compute_predict_proba(self, input_data):
        return self.model.predict_proba(input_data)
# =============================================================================
# # 5) Define a função para construir um modelo para cada palavra
# =============================================================================
def build_models(input_folder):
    # Inicializa a variável para armazenar todos os modelos
    speech_models = []

    # 6) Parse no diretório de entrada
    for dirname in os.listdir(input_folder):
        # Pega o nome do subdiretório
        subfolder = os.path.join(input_folder, dirname)

        if not os.path.isdir(subfolder):
            continue

        # Extrai o label=nome-da-pasta
        label = subfolder[subfolder.rfind('/') + 1:]

        # Inicializa a variável do tipo NumPy
        X = np.array([])

        # Cria uma lista de arquivos para serem usados no treinamento
        # Será deixado um arquivo por pasta para servir de teste
        training_files = [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]

# =============================================================================
#       # 7) Iterar os arquivos de treinamento e construir os modelos
# =============================================================================
        for filename in training_files:
            # Extrair o atual filepath
            filepath = os.path.join(subfolder, filename)

            # Ler o sinal áudio do arquivo de entrada
            sampling_freq, signal = wavfile.read(filepath)

# =============================================================================
#           # 8) Extrair os MFCC features
# =============================================================================
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                features_mfcc = mfcc(signal, sampling_freq, nfft=1200)

            # Atribui à varável X
            if len(X) == 0:
                X = features_mfcc
            else:
                X = np.append(X, features_mfcc, axis=0)

# =============================================================================
#       # 9) Criar o modelo HMM
# =============================================================================
        model = ModelHMM()

        # 9) Treinar o HMM
        model.train(X)

        # 9) Salvar o modelo para a palavra atual
        speech_models.append((model, label))
        
        #Mostrar o valor predito
        

        # 9) Resetar a variável
        model = None

    return speech_models

# =============================================================================
# # 10) Definir uma função para executar testes em arquivos de entrada
# =============================================================================
def run_tests(test_files):
    # Classificar os dados de entrada
    nro = 0
    for test_file in test_files:
        # Ler arquivos de entrada
        sampling_freq, signal = wavfile.read(test_file)

        # Extrair MFCC features
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            features_mfcc = mfcc(signal, sampling_freq, nfft= 1200)

        # Definir variáveis
        max_score = -float('inf')
        max_predi = ''
        max_predi_proba = ''
        
        #output_label = None # variável original

#=============================================================================
#         # 11) Executar o vetor numpy de features atual por todo o HMM
#         # Escolher o modelo que tiver a maior pontuação
#=============================================================================
        #max_score_label = ([],[])
        for item in speech_models:
            model, label = item
            score = model.compute_score(features_mfcc)
            if score > max_score:
                max_score = score
                predicted_label = label
                max_predi = model.compute_predict(features_mfcc)
                max_predi_proba = model.compute_predict_proba(features_mfcc)

        # Imprimir a saída que foi estimada
        start_index = test_file.find('/') + 1
        end_index = test_file.rfind('/')
        original_label = test_file[start_index:end_index]
        print('Score: ')
        print(max_score)
        
        print('Predito: ')
        print(max_predi)
        
        print('Predito proba: ')
        print(max_predi_proba)
        
        print('\nOriginal:', original_label)
        print('Estimado:', predicted_label)

#%%
#==============================================================================
# 12) Faz com que o conjunto de funções e treinamento 
#==============================================================================
import pickle
#from sklearn.externals import joblib

if __name__=='__main__':
    #args = build_arg_parser().parse_args()
    #input_folder = args.input_folder
    input_folder = '/home/marcello/Documentos/work/Ministerio-Cidadania/01-transcritor/data/'

    # Criar um modelo HMM para cada palavra
    speech_models = build_models(input_folder)
    
    save_model = 'transcritor.pkl'
    with open(save_model, "wb") as file: pickle.dump(speech_models, save_model)

#%%
#==============================================================================
# 13) Arquivos para teste: corresponde a um dos arquivos de cada subpasta
#==============================================================================
    test_files = []
    for root, dirs, files in os.walk(input_folder):
        for filename in (x for x in files if '6' in x):
            filepath = os.path.join(root, filename)
            test_files.append(filepath)

    # Executar o teste
    run_tests(test_files)

#%%
with open(save_model, "rb") as file: 
    load_model = pickle.load(save_model)

file_test = '/home/marcello/Documentos/work/Ministerio-Cidadania/01-transcritor/data-test/ministerio-da-cidadania.wav'
sampling_freq, signal = wavfile.read(file_test)
features_mfcc = mfcc(signal, sampling_freq, nfft=1200)
r_score       = load_model.compute_score(features_mfcc)
r_predict_proba = load_model.compute_predict_proba(features_mfcc)
print('score: ')
print(r_score)
print('r_predict_proba: ')
print(r_predict_proba)
#COLOCAR AQUI O CÓDIGO PARA GRAVAR O predicted_label NO ARQUIVO
# Gerar um gráfico contendo o nro da iteracao x max_score
#arq = 'arq'+str(nro)+'.txt'
arq = 'arquivo.txt'
with open(arq) as arquivo:
    print(arquivo.read())
