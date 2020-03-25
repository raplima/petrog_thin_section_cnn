Python scripts used to perform petrographyc thin section analysis using convolutional neural networks published as:
## Petrographic microfacies classification with deep convolutional neural networks
Journal title: Computers and Geosciences  
Authors: Rafael Pires de Lima, David Duarte, Charles Nicholson, Roger Slatt, Kurt J. Marfurt  
Corresponding author: Rafael Pires de Lima  

## Scripts
- simple_cb.py: performs color balancing. **Please note the original source and references for the script and algorithm. **

- cnn_figs.py:	creates (completely or partially) figures used for the paper
- cnn_processing.py: functions to fine tune CNNs
- data_manipulation.py:	data preparation
- cnn_evaluate.py: used to evaluate of CNN models generated with cnn_processing.py
- test_pub_data.py:	functions for the evaluation of public data
- metrics_and_confusion_matrix_plot.R:	R files for metrics and confusion plots
- metrics_and_confusion_matrix_plot_for_public.R:	R files for metrics and confusion plots

For an easier to use tool for initial transfer learning evaluation, an user iterface is provided in https://github.com/raplima/transfer_learning_wgui. 

_Software here is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied._
