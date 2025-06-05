# IBTransDiff

This repository provides an implementation of a diffusion-based time series forecasting framework guided by Transformer-derived global context and regularized via Information Bottleneck principles.

## 1. Setup
Install required packages:
```bash
pip install -r requirements.txt
```

Environment:
* Python 3.8
* Pytorch 2.1.2
* Ubuntu 20.04


## 2. Dataset
The datasets can be obtained from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)

## 3. Run Experiment

### Full pipeline
```
python run.py
```
### Only test
```
python run.py --is_training False
```

## 4. Pretrained Model
Pretrained weights used to generate the reported results are available in the \texttt{checkpoints/} directory of the repository.

## 5. Experiment Results
\begin{table*}[t]
    \centering
        \caption{\textbf{Quantitative comparison in terms of MSE and MAE} across five real-world datasets for Transformer-based models (T), diffusion-based models (D), Information Bottleneck-based models (I), and our method. The best and second-best performance are highlited in \textcolor{WildStrawberry!90}{red} and \textcolor{Aquamarine!80}{blue}. \textbf{Ours$^{\text{N}}$} and  \textbf{Ours$^{\text{I}}$} denote our IB-TransDiff combined with NSTransformer~\cite{Stationary} and Informer~\cite{zhou2021informer}.}
    \label{tab:main_results}
    \vspace{-2mm}
    \setlength{\tabcolsep}{5pt}
    \scriptsize
    \resizebox{\textwidth}{!}{
    \begin{tabular}{ll|cc|cc|cc|cc|cc}
        \toprule
        \multicolumn{2}{l}{Dataset} & \multicolumn{2}{c}{Exchange} & \multicolumn{2}{c}{ILI} &
        \multicolumn{2}{c}{ETTm2} & \multicolumn{2}{c}{Electricity} & \multicolumn{2}{c}{Weather} \\
        \midrule
        \multicolumn{2}{l|}{Method} & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE \\
        \midrule
        \multirow{1}{*}{Transformer (T)\cite{vaswani2017attentionisallyouneed}} & & 
        1.20$_{\pm0.129}$ & 0.84$_{\pm0.041}$ & 4.93$_{\pm0.277}$ & 1.48$_{\pm0.082}$ & 
        4.76$_{\pm1.214}$ & 1.76$_{\pm0.256}$ & 
        0.26$_{\pm0.016}$ & 0.36$_{\pm0.015}$ & 
        0.57$_{\pm0.045}$ & 0.53$_{\pm0.024}$\\
        \midrule
        \multirow{1}{*}{Informer (T)\cite{zhou2021informer}} & & 
        1.31$_{\pm0.154}$ & 0.85$_{\pm0.021}$ & 5.33$_{\pm0.177}$ & 1.59$_{\pm0.078}$ & 
        5.74$_{\pm0.475}$ & 1.99$_{\pm0.181}$ & 
        0.35$_{\pm0.012}$ & 0.43$_{\pm0.020}$ & 
        0.48$_{\pm0.078}$ & 0.47$_{\pm0.045}$\\
        \midrule
        \multirow{1}{*}{Autoformer (T)\cite{chen2021autoformer}} & & 
        0.44$_{\pm0.146}$ & 0.48$_{\pm0.082}$ & 3.26$_{\pm0.180}$ & 1.25$_{\pm0.066}$ & 
        0.47$_{\pm0.032}$ & 0.46$_{\pm0.021}$ & 
        0.22$_{\pm0.019}$ & 0.33$_{\pm0.015}$ & 
        0.31$_{\pm0.018}$ & 0.37$_{\pm0.025}$\\
        \midrule
        \multirow{1}{*}{NSTransformer (T)\cite{Stationary}} & & 
        0.25$_{\pm0.088}$ & 0.36$_{\pm0.091}$ & \color{Aquamarine!80}\textbf{1.93}$_{\pm0.157}$ & 0.87$_{\pm0.058}$ & 
        0.53$_{\pm0.042}$ & 0.48$_{\pm0.016}$ & 
        0.18$_{\pm0.012}$ & 
        0.28$_{\pm0.012}$ & 
        0.25$_{\pm0.016}$ & 0.29$_{\pm0.020}$\\
        \midrule
        \multirow{1}{*}{TimeGrad (D)\cite{rasul2021timegrad}} & & 
        2.43$_{\pm0.229}$ & 0.90$_{\pm0.232}$ &
        2.65$_{\pm0.164}$ & 1.15$_{\pm0.172}$ & 1.36$_{\pm0.133}$ & 0.74$_{\pm0.123}$ & 0.69$_{\pm0.188}$ & 0.74$_{\pm0.109}$ & 0.90$_{\pm0.139}$ & 0.57$_{\pm0.136}$ \\
        \midrule
        \multirow{1}{*}{CSDI (D)\cite{tashiro2021csdi}} & & 
        1.67$_{\pm0.162}$ & 0.75$_{\pm0.058}$ & 
        2.54$_{\pm0.098}$ & 1.21$_{\pm0.128}$ & 
        1.28$_{\pm0.074}$ & 0.67$_{\pm0.064}$ & 
        0.56$_{\pm0.212}$ & 0.81$_{\pm0.150}$ & 
        0.86$_{\pm0.073}$ & 0.56$_{\pm0.096}$ \\
        \midrule
        \multirow{1}{*}{SSSD (D)\cite{alcaraz2022diffusion}} & & 
        0.90$_{\pm0.171}$ & 0.86$_{\pm0.127}$ & 
        2.52$_{\pm0.118}$ & 1.08$_{\pm0.131}$ & 
        0.97$_{\pm0.043}$ & 0.56$_{\pm0.060}$ & 
        0.47$_{\pm0.129}$ & 0.60$_{\pm0.207}$ & 
        0.67$_{\pm0.159}$ & 0.49$_{\pm0.106}$ \\
        \midrule
        \multirow{1}{*}{D$^3$VAE (D)\cite{li2022generative}} & & 
        0.76$_{\pm0.118}$ & 0.62$_{\pm0.108}$ & 2.44$_{\pm0.115}$ & 1.11$_{\pm0.127}$ & 
        0.79$_{\pm0.038}$ & 0.46$_{\pm0.047}$ & 
        0.33$_{\pm0.194}$ & 0.49$_{\pm0.119}$ & 
        0.43$_{\pm0.139}$ & 0.34$_{\pm0.131}$ \\
        \midrule
        \multirow{1}{*}{TimeDiff (D)\cite{shen2023timediff}} & & 
        0.48$_{\pm0.095}$ & 0.43$_{\pm0.109}$ & 2.46$_{\pm0.148}$ & 1.09$_{\pm0.064}$ & 0.41$_{\pm0.014}$ & 0.42$_{\pm0.013}$ & 
        0.27$_{\pm0.024}$ & 0.32$_{\pm0.131}$ & 
        0.36$_{\pm0.146}$ & 0.37$_{\pm0.052}$ \\
        \midrule
        \multirow{1}{*}{TMDM (D)\cite{transformermodulate}} & & 
        0.26$_{\pm0.019}$ & 0.37$_{\pm0.015}$ & 
        1.99$_{\pm0.085}$ & \color{Aquamarine!80}\textbf{0.85}$_{\pm0.026}$ & 
        0.27$_{\pm0.023}$ & 0.35$_{\pm0.015}$ & 
        0.19$_{\pm0.007}$ & 0.27$_{\pm0.008}$ & 
        0.28$_{\pm0.095}$ & \color{Aquamarine!80}\textbf{0.25}$_{\pm0.103}$\\
        \midrule
        \multirow{1}{*}{TimeSieve (I)\cite{feng2024timesieve}} & & 
        0.30$_{\pm0.049}$ & 0.40$_{\pm0.029}$ & 
        3.62$_{\pm0.080}$ & 1.32$_{\pm0.017}$ & 
        0.82$_{\pm1.077}$ & 0.54$_{\pm0.391}$ & 
        \color{Aquamarine!80}\textbf{0.18}$_{\pm0.000}$ & \color{Aquamarine!80}\textbf{0.27}$_{\pm0.000}$ & 
        0.26$_{\pm0.126}$ & 0.31$_{\pm0.102}$\\
        \midrule
        \multirow{1}{*}{\textbf{Ours$^{\text{N}}$}} & &  
        \color{Aquamarine!80}\textbf{0.21}$_{\pm0.004}$ &  \color{Aquamarine!80}\textbf{0.33}$_{\pm0.004}$ &
        2.72$_{\pm0.143}$ & 1.03$_{\pm0.018}$ & \color{Aquamarine!80}\textbf{0.26}$_{\pm0.005}$ & \color{Aquamarine!80}\textbf{0.31}$_{\pm0.004}$ & 
        0.19$_{\pm0.001}$ & 
        0.29$_{\pm0.001}$ & 
        \color{Aquamarine!80}\textbf{0.23}$_{\pm0.004}$ & 
        0.28$_{\pm0.004}$ \\
        \midrule
        \multirow{1}{*}{\textbf{Ours$^{\text{I}}$}} & & 
        \color{WildStrawberry!90}\textbf{0.16}$_{\pm0.041}$ &  \color{WildStrawberry!90}\textbf{0.29}$_{\pm0.037}$ & 
        \color{WildStrawberry!90}\textbf{1.64}$_{\pm0.087}$ & 
        \color{WildStrawberry!90}\textbf{0.82}$_{\pm0.018}$ & 
        \color{WildStrawberry!90}\textbf{0.04}$_{\pm0.011}$ & 
        \color{WildStrawberry!90}\textbf{0.15}$_{\pm0.018}$ & 
        \color{WildStrawberry!90}\textbf{0.10}$_{\pm0.007}$ & 
        \color{WildStrawberry!90}\textbf{0.23}$_{\pm0.008}$ & 
        \color{WildStrawberry!90}\textbf{0.003}$_{\pm0.000}$ & 
        \color{WildStrawberry!90}\textbf{0.04}$_{\pm0.003}$ \\
        \bottomrule
    \end{tabular}%
    }
\end{table*}

