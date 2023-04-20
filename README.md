# **CPSC 5616**: Robot Modelling by Using LSTMs

### Description
Group Project of CPSC5616 Machine and Deep Learning at Laurentian University

### Group Member
1. [Wang, Pengyu](https://github.com/PengyuW007), 0425157
2. [Zhang, Haokun](https://github.com/haokunzhang), 0424660
3. [Zhu, Ziping](https://github.com/0v0-QAQ) or [Zhu, Ziping](https://github.com/zzhu4LU), 0422426

### Version Control Systems
1. [Repository](https://github.com/PengyuW007/CPSC5616_Robot_Modelling.git) OR	

	gh repo clone PengyuW007/CPSC5616_Robot_Modelling

2. [Google Drive](https://drive.google.com/drive/folders/1sdCNKh6dScgluBZTkN8jLNDVo_9A2goP?ths=true)

### Compile and Run
1. Whatever you run on any machine, please make sure you have matlab compiler

2. MatLab Version: R2022a

3. If you have MatLab compiler, just click run button. 
Please run these three files, MLP.m under src/MLP_Sequential_sigmoid, project_v2.m under src/MLP_Batch_tanh and LSTM.m under src/LSTM folders, which shows the results of MLP and LSTM after training.

Reminder: 

    1. MLP.m -> src/MLP_Sequential_sigmoid: This would take 2 MINs.

    2. project_v2.m -> src/MLP_Batch_tanh: At line 50, there you can adjust the number of epoch, if you want to show complete results matches with report, 
you can adjust it to 50000, which took us 1 hour by CPU: i7-10750H OR 30 min by Apple M1.
    If you think it is too long to wait, decrease it to 2000 would be better for you.
 
    3. LSTM.m -> src/LSTM: It took 2 mins.

### Datasets
There are 3 files for input and output, for training convenience, I separate them into 3. Please run the third one
" Dataset_with_6 inputs and 2 Outputs.xlsx", which contains the whole dataset.
    
1. Dataset_5000.xlsx: This file contains 5000 rows of data.
    
2. Dataset_300000.xlsx:This file contains 349356 rows of data.
    
3. Dataset_with_6 inputs and 2 Outputs.xlsx (Robot Dataset_with_6 inputs and 2 Outputs.xlsx): This file contains 698710 rows of data. And this is the original file.

### Packages:
    - Paper
        - IEEE Paper Format.docx
        - Robot Modelling Using LSTMs and BP.docx: Final report for word format.
        - Robot Modelling Using LSTMs and BP.docx.pdf: Final report for pdf format
    - Presentation
        - gifs used on presentation slides
        - CPSC 5616 Robot Modelling Using LSTMs_V1_Presentation.pdf
        - CPSC 5616 Robot Modelling Using LSTMs_V2_Presentation.pdf
        - CPSC 5616 Robot Modelling Using LSTMs_V3_Presentation.pdf: Final version for presentation
    - res: references
        - Papers
            - Collaborative_Human-Robot_Motion_Generation_Using_LSTM-RNN.pdf
            - Deep Learning-based Robot Control using Recurrent Neural Networks (LSTM).pdf
            - Optimization_and_improvement_o.pdf
            - Trajectory Prediction of Surrounding Vehicles Using LSTM Network.pdf
        - Slides
            - CMU_RNN.pdf
            - StandfordU_CS224n-2023rnn.pdf
    - src: Codes
        - LSTM
            - Dataset_5000.xlsx: This file contains 5000 rows of data.
            - Dataset_300000.xlsx:This file contains 349356 rows of data.
            - Dataset_with_6 inputs and 2 Outputs.xlsx (Robot Dataset_with_6 inputs and 2 Outputs.xlsx): This file contains 698710 rows of data. And this is the original file.
            - LSTM.m
        - MLP_Batch_tanh
            - Dataset_5000.xlsx: This file contains 5000 rows of data.
            - Dataset_300000.xlsx:This file contains 349356 rows of data.
            - Dataset_with_6 inputs and 2 Outputs.xlsx (Robot Dataset_with_6 inputs and 2 Outputs.xlsx): This file contains 698710 rows of data. And this is the original file.
            - project_v2.m: Code for running MLP's batch training
            - output.fig
            - trainedWeights_300000data_50000epoch_0.000001LR.mat: weights value after 50000 epoch
        - MLP_Sequential_sigmoid
            - Dataset_5000.xlsx: This file contains 5000 rows of data.
            - Dataset_300000.xlsx:This file contains 349356 rows of data.
            - Dataset_with_6 inputs and 2 Outputs.xlsx (Robot Dataset_with_6 inputs and 2 Outputs.xlsx): This file contains 698710 rows of data. And this is the original file.
            - MLP.m: Code for running MLP's sequential training.
            - MLPError.xlsx: Result of MLP's error
            - MLPweightsData.xlsx
    - .gitignore
    - Meeting notes

### Additional notes
These files run over 698710/300000+ rows of data. All the graphs and plots show up, which depends on the performance of your CPU and Memory. So, these files would take a while to run, just **BE PATIENT**! Much appreciate for your patience!
