# CTR-Estimation
An index of recommendation algorithms about CTR-Estimation.

Our survey **[Deep Learning for Click-Through Rate Estimation](https://arxiv.org/pdf/2104.10584)** is available.

Please cite our survey paper if this index is helpful.
```
@article{zhang2021deep,
  title={Deep learning for click-through rate estimation},
  author={Zhang, Weinan and Qin, Jiarui and Guo, Wei and Tang, Ruiming and He, Xiuqiang},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, {IJCAI} 2021, Virtual Event / Montreal, Canada, 19-27 August 2021},
  pages = {4695--4703},
  year={2021}
}
```

### Shallow CTR Models
| **Name** | **Paper** | **Venue** | **Year** | **Code** |
| --- | --- | --- | --- | --- |
| LR | [Richardson, M., Dominowska, E., & Ragno, R. (2007, May). Predicting clicks: estimating the click-through rate for new ads. In Proceedings of the 16th international conference on World Wide Web (pp. 521-530).](http://www2007.org/papers/paper784.pdf) | WWW | 2007 | NA |
| POLY2 | [Chang, Y. W., Hsieh, C. J., Chang, K. W., Ringgaard, M., & Lin, C. J. (2010). Training and testing low-degree polynomial data mappings via linear SVM. Journal of Machine Learning Research, 11(4).](https://www.jmlr.org/papers/volume11/chang10a/chang10a.pdf) | JMLR | 2010 | NA |
| GBDT | [He, X., Pan, J., Jin, O., Xu, T., Liu, B., Xu, T., ... & Candela, J. Q. (2014, August). Practical lessons from predicting clicks on ads at facebook. In Proceedings of the eighth international workshop on data mining for online advertising (pp. 1-9).](http://quinonero.net/Publications/predicting-clicks-facebook.pdf) | ADKDD | 2014 | NA |
| FM | [Rendle, S. (2010, December). Factorization machines. In 2010 IEEE International conference on data mining (pp. 995-1000). IEEE.](https://analyticsconsultores.com.mx/wp-content/uploads/2019/03/Factorization-Machines-Steffen-Rendle-Osaka-University-2010.pdf) | ICDM | 2010 | NA |
| FFM | [Juan, Y., Zhuang, Y., Chin, W. S., & Lin, C. J. (2016, September). Field-aware factorization machines for CTR prediction. In Proceedings of the 10th ACM conference on recommender systems (pp. 43-50).](https://www.andrew.cmu.edu/user/yongzhua/conferences/ffm.pdf) | RecSys | 2016 | [Python](https://paperswithcode.com/paper/field-aware-factorization-machines-for-ctr) |
| FwFM | [Pan, J., Xu, J., Ruiz, A. L., Zhao, W., Pan, S., Sun, Y., & Lu, Q. (2018, April). Field-weighted factorization machines for click-through rate prediction in display advertising. In Proceedings of the 2018 World Wide Web Conference (pp. 1349-1357).](https://dl.acm.org/doi/pdf/10.1145/3178876.3186040) | WWW | 2018 | [Python](https://paperswithcode.com/paper/field-weighted-factorization-machines-for) |
| LorentzFM | [Xu, C., & Wu, M. (2020, April). Learning feature interactions with lorentzian factorization machine. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 04, pp. 6470-6477).](https://ojs.aaai.org/index.php/AAAI/article/download/6119/5975) | AAAI | 2020 | [Python](https://paperswithcode.com/paper/learning-feature-interactions-with-lorentzian) |
| FM$^{2}$ | [Sun, Y., Pan, J., Zhang, A., & Flores, A. (2021, April). Fm2: Field-matrixed factorization machines for recommender systems. In Proceedings of the Web Conference 2021 (pp. 2828-2837).](https://arxiv.org/pdf/2102.12994) | WWW | 2021 | [Python](https://paperswithcode.com/paper/fm-2-field-matrixed-factorization-machines) |








### Feature Interaction via DNN
| **Name** | **Paper** | **Venue** | **Year** | **Code** |
| --- | --- | --- | --- | --- |
| PNN | [Qu, Y., Cai, H., Ren, K., Zhang, W., Yu, Y., Wen, Y., & Wang, J. (2016, December). Product-based neural networks for user response prediction. In 2016 IEEE 16th International Conference on Data Mining (ICDM) (pp. 1149-1154). IEEE.](https://arxiv.org/pdf/1611.00144) | ICDM | 2016 | [Python](https://paperswithcode.com/paper/product-based-neural-networks-for-user) |
| Wide&Deep | [Cheng, H. T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., ... & Shah, H. (2016, September). Wide & deep learning for recommender systems. In Proceedings of the 1st workshop on deep learning for recommender systems (pp. 7-10).](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454) | DLRS | 2016 | [Python](https://paperswithcode.com/paper/wide-deep-learning-for-recommender-systems) |
| Deep&Cross | [Wang, R., Fu, B., Fu, G., & Wang, M. (2017). Deep & cross network for ad click predictions. In Proceedings of the ADKDD'17 (pp. 1-7).](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754) | ADKDD | 2017 | [Python](https://paperswithcode.com/paper/deep-cross-network-for-ad-click-predictions) |
| FNN | [Zhang, W., Du, T., & Wang, J. (2016, March). Deep learning over multi-field categorical data. In European conference on information retrieval (pp. 45-57). Springer, Cham.](https://arxiv.org/pdf/1601.02376.pdf) | ECIR | 2016 | [Python](https://paperswithcode.com/paper/deep-learning-over-multi-field-categorical) |
| DeepFM | [Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017, August). DeepFM: a factorization-machine based neural network for CTR prediction. In Proceedings of the 26th International Joint Conference on Artificial Intelligence (pp. 1725-1731).](https://arxiv.org/pdf/1703.04247.pdf)| ICJAI | 2017 | [Python](https://paperswithcode.com/paper/deepfm-a-factorization-machine-based-neural) |
| NFM | [He, X., & Chua, T. S. (2017, August). Neural factorization machines for sparse predictive analytics. In Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval (pp. 355-364).](https://arxiv.org/pdf/1708.05027.pdf) | SIGIR | 2017 | [Python](https://paperswithcode.com/paper/neural-factorization-machines-for-sparse) |
| xDeepFM | [Lian, J., Zhou, X., Zhang, F., Chen, Z., Xie, X., & Sun, G. (2018, July). xdeepfm: Combining explicit and implicit feature interactions for recommender systems. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 1754-1763).](https://arxiv.org/pdf/1803.05170.pdf) | SIGKDD | 2018 | [Python](https://github.com/Leavingseason/xDeepFM)
| AFM | [Xiao, J., Ye, H., He, X., Zhang, H., Wu, F., & Chua, T. S. (2017). Attentional factorization machines: Learning the weight of feature interactions via attention networks. arXiv preprint arXiv:1708.04617.](https://arxiv.org/pdf/1708.04617.pdf?ref=https://git.chanpinqingbaoju.com) | ICJAI | 2017 | [Python](https://paperswithcode.com/paper/attentional-factorization-machines-learning) |
| FiBiNET | [Huang, T., Zhang, Z., & Zhang, J. (2019, September). FiBiNET: combining feature importance and bilinear feature interaction for click-through rate prediction. In Proceedings of the 13th ACM Conference on Recommender Systems (pp. 169-177).](https://arxiv.org/pdf/1905.09433.pdf?ref=https://githubhelp.com) | RecSys | 2019 | [Python](https://paperswithcode.com/paper/fibinet-combining-feature-importance-and) |
| OENN | [Guo, W., Tang, R., Guo, H., Han, J., Yang, W., & Zhang, Y. (2019, July). Order-aware embedding neural network for CTR prediction. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 1121-1124).](https://www.researchgate.net/profile/Huifeng-Guo/publication/334586159_Order-aware_Embedding_Neural_Network_for_CTR_Prediction/links/5e8e75ec299bf1307989da0b/Order-aware-Embedding-Neural-Network-for-CTR-Prediction.pdf) | SIGIR | 2019 |  |
| DCN V2 | [Wang, R., Shivanna, R., Cheng, D., Jain, S., Lin, D., Hong, L., & Chi, E. (2021, April). DCN V2: Improved deep & cross network and practical lessons for web-scale learning to rank systems. In Proceedings of the Web Conference 2021 (pp. 1785-1797).](https://arxiv.org/pdf/2008.13535.pdf?ref=https://codemonkey.link) | WWW | 2021 | [Python](https://paperswithcode.com/paper/dcn-m-improved-deep-cross-network-for-feature) |

### Automatic Feature Interaction
| **Name** | **Paper** | **Venue** | **Year** | **Code** |
| --- | --- | --- | --- | --- |
| AutoInt | [Song, W., Shi, C., Xiao, Z., Duan, Z., Xu, Y., Zhang, M., & Tang, J. (2019, November). Autoint: Automatic feature interaction learning via self-attentive neural networks. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management (pp. 1161-1170).](https://arxiv.org/pdf/1810.11921) | CIKM | 2019 | [Python](https://paperswithcode.com/paper/autoint-automatic-feature-interaction) |
| AFN | [Cheng, W., Shen, Y., & Huang, L. (2020, April). Adaptive factorization network: Learning adaptive-order feature interactions. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 04, pp. 3609-3616).](https://ojs.aaai.org/index.php/AAAI/article/view/5768/5624) | AAAI | 2020 | [Python](https://paperswithcode.com/paper/adaptive-factorization-network-learning) |
| AutoFIS | [Liu, B., Zhu, C., Li, G., Zhang, W., Lai, J., Tang, R., ... & Yu, Y. (2020, August). Autofis: Automatic feature interaction selection in factorization models for click-through rate prediction. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2636-2645).](https://arxiv.org/pdf/2003.11235) | SIGKDD | 2020 | [Python](https://paperswithcode.com/paper/autofis-automatic-feature-interaction) |
| AIM | [Zhu, C., Chen, B., Zhang, W., Lai, J., Tang, R., He, X., ... & Yu, Y. (2021). AIM: Automatic Interaction Machine for Click-Through Rate Prediction. IEEE Transactions on Knowledge and Data Engineering.](https://arxiv.org/pdf/2111.03318) | TKDE | 2021 | [Python](https://paperswithcode.com/paper/aim-automatic-interaction-machine-for-click) |



### Feature Interactions via GNN
| **Name** | **Paper** | **Venue** | **Year** | **Code** |
| --- | --- | --- | --- | --- |
| Fi-GNN | [Li, Z., Cui, Z., Wu, S., Zhang, X., & Wang, L. (2019, November). Fi-gnn: Modeling feature interactions via graph neural networks for ctr prediction. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management (pp. 539-548).](https://arxiv.org/pdf/1910.05552.pdf?ref=https://githubhelp.com) | CIKM | 2019 | [Python](https://paperswithcode.com/paper/fi-gnn-modeling-feature-interactions-via) |
| $L_{0}$-SIGN | [Su, Y., Zhang, R., Erfani, S., & Xu, Z. (2021, February). Detecting beneficial feature interactions for recommender systems. In Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI).](https://www.aaai.org/AAAI21Papers/AAAI-279.SuY.pdf) | AAAI | 2021 | [Python](https://paperswithcode.com/paper/detecting-relevant-feature-interactions-for) |
| PCF-GNN | [Li, F., Yan, B., Long, Q., Wang, P., Lin, W., Xu, J., & Zheng, B. (2021, July). Explicit Semantic Cross Feature Learning via Pre-trained Graph Neural Networks for CTR Prediction. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 2161-2165).](https://arxiv.org/pdf/2105.07752) | SIGIR | 2021 | [Python](https://paperswithcode.com/paper/explicit-semantic-cross-feature-learning-via) |
| DG-ENN | [Guo, W., Su, R., Tan, R., Guo, H., Zhang, Y., Liu, Z., ... & He, X. (2021, August). Dual Graph enhanced Embedding Neural Network for CTR Prediction. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (pp. 496-504).](https://arxiv.org/pdf/2106.00314) | SIGKDD | 2021 | NA |

### Sequential Feature Interactions
| **Name** | **Paper** | **Venue** | **Year** | **Code** |
| --- | --- | --- | --- | --- |
| DIN | [Zhou, G., Zhu, X., Song, C., Fan, Y., Zhu, H., Ma, X., ... & Gai, K. (2018, July). Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 1059-1068).](https://arxiv.org/pdf/1706.06978) | SIGKDD | 2018 | [Python](https://paperswithcode.com/paper/deep-interest-network-for-click-through-rate) |
| DIEN | [Zhou, G., Mou, N., Fan, Y., Pi, Q., Bian, W., Zhou, C., ... & Gai, K. (2019, July). Deep interest evolution network for click-through rate prediction. In Proceedings of the AAAI conference on artificial intelligence (Vol. 33, No. 01, pp. 5941-5948).](https://ojs.aaai.org/index.php/AAAI/article/download/4545/4423) | AAAI | 2019 | [Python](https://paperswithcode.com/paper/deep-interest-evolution-network-for-click) |
| DSIN | [Feng, Y., Lv, F., Shen, W., Wang, M., Sun, F., Zhu, Y., & Yang, K. (2019). Deep session interest network for click-through rate prediction. arXiv preprint arXiv:1905.06482.](https://arxiv.org/pdf/1905.06482) | ICJAI | 2019 | [Python](https://paperswithcode.com/paper/deep-session-interest-network-for-click) |
| DMR | [Lyu, Z., Dong, Y., Huo, C., & Ren, W. (2020, April). Deep match to rank model for personalized click-through rate prediction. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 01, pp. 156-163).](https://ojs.aaai.org/index.php/AAAI/article/view/5346/5202) | AAAI | 2020 | [Python](https://github.com/lvze92/DMR) |
| CAN | [Bian, W., Wu, K., Ren, L., Pi, Q., Zhang, Y., Xiao, C., ... & Deng, H. (2022, February). CAN: Feature Co-Action Network for Click-Through Rate Prediction. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining (pp. 57-65).](https://dl.acm.org/doi/abs/10.1145/3488560.3498435) | WSDM | 2022 | [Python](https://github.com/CAN-Paper/Co-Action-Network) |
