Python and R scripts used to perform petrographyc thin section classification using convolutional neural networks as was published in:
## Petrographic microfacies classification with deep convolutional neural networks
Journal title: Computers and Geosciences  
Authors: Rafael Pires de Lima, David Duarte, Charles Nicholson, Roger Slatt, Kurt J. Marfurt  
Corresponding author: Rafael Pires de Lima  

```bibtex
@article
{ 
  PIRESDELIMA2020104481,
  title = "Petrographic microfacies classification with deep convolutional neural networks",
  journal = "Computers & Geosciences",
  volume = "142",
  pages = "104481",
  year = "2020",
  issn = "0098-3004",
  doi = "https://doi.org/10.1016/j.cageo.2020.104481",
  url = "http://www.sciencedirect.com/science/article/pii/S0098300419307629",
  author = "Rafael {Pires de Lima} and David Duarte and Charles Nicholson and Roger Slatt and Kurt J. Marfurt",
  keywords = "Petrography thin section analysis, Rock thin section, Convolutional neural networks, Transfer learning",
}
```

## Data

Data can be downloaded [here](https://data.mendeley.com/datasets/vsnhrtdx22/draft?a=6a5b599e-781f-4dc7-aba0-72a03ae540ae). 

## Scripts
- [`simple_cb.py`](./simple_cb.py): performs color balancing. **Please note the original source and references for the script and algorithm. **
- [`cnn_figs.py`](./cnn_figs.py):	creates (completely or partially) figures used for the paper
- [`cnn_processing.py`](./cnn_processing.py): functions to fine tune CNNs
- [`data_manipulation.py`](./data_manipulation.py):	data preparation
- [`cnn_evaluate.py`](./cnn_evaluate.py): used to evaluate of CNN models generated with cnn_processing.py
- [`test_pub_data.py`](./test_pub_data.py):	functions for the evaluation of public data
- [`metrics_and_confusion_matrix_plot.R`](./metrics_and_confusion_matrix_plot.R):	R files for metrics and confusion plots
- [`metrics_and_confusion_matrix_plot_for_public.R`](./metrics_and_confusion_matrix_plot_for_public.R):	R files for metrics and confusion plots

For an easier to use tool for initial transfer learning evaluation, an user iterface is provided [here](https://github.com/raplima/transfer_learning_wgui). 

_Software here is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied._
