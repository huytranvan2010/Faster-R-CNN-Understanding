# Faster-R-CNN-Understanding

## Giới thiệu
Trong bài trước chúng ta đã tìm hiểu về mô hình [Fast R-CNN](https://huytranvan2010.github.io/Fast-R-CNN-Understanding/). Mô hình Fast R-CNN đã có cải tiến rất nhiều so với R-CNN nhưng nó vẫn có bottleneck với region proposals làm hạn chế performance. Cũng giống như R-CNN, Fast R-CNN vẫn còn sử dụng Selective Search để tạo ra các region proposals. Mô hình Faster R-CNN có giới thiệu *Region Proposal Network (RPN)* để tạo ra các region proposals, FPN chia sẻ convolutional features với phần detection (chính là Fast R-CNN). Chính điều này đã giúp Faster R-CNN chính xác hơn và có performance tốt hơn so với 2 người anh em của mình.

## Architecture
Faster R-CNN có kiến trúc khá phức tạp, nó được tạo thành từ một số phần. Chúng ta sẽ đi tổng quan trước, sau đó sẽ đi vào chi tiết các thành phần.

Đối với bài toán object detection chúng ta muốn:
- Danh sách các bounding boxes
- Label tương ứng với mỗi bounding box
- Confidence score cho mỗi bounding box

<img src="https://tryolabs.com/blog/images/blog/post-images/2018-01-18-faster-rcnn/fasterrcnn-architecture.b9035cba.png" style="display:block; margin-left:auto; margin-right:auto">

*Kiến trúc của Faster R-CNN*

Ảnh ban đầu đi qua pre-trained CNN để trích xuất convolutional feature map. Feature map này sẽ được đưa vào mạng RPN (Region proposal Network). RPN là mạng fully convolution network, đây là một mạng chứa toàn Conv layers và khá đơn giản. RPN sẽ giúp chúng ta tạo ra các region proposals. Để thực hiện việc này Faster R-CNN đưa vào một khái niệm là **anchor boxes** - đây là những box được định nghĩa sẵn. Nhiệm vụ của RPN đi xác định xem các anchor boxes có chứa object hay không và tinh chỉnh vị trí của nó sao cho khớp với ground-truth box nhất.

Đầu ra của RPN là các region proposals. Tương tự như Fast R-CNN, khi biết được vị trí và kích thước của region proposals trên ảnh ban đầu, chúng ta có thể xác định phần feature map tương ứng của nó trên feature map. Phần feature map này đi qua RoI pooling layer để đưa về kích thước cố định rồi đi phân loại và tinh chỉnh lại vị trí của bounding box.

## Base Network
Đầu tiên chúng ta cần sử dụng pre-trained CNN model để có được feature map. Pre-trained CNN model này thường được train trên bộ dữ liệu ImageNet. Trong bài báo gốc tác giả sử dựng model **ZF** và **VGG**. Tuy nhiên hiện tại có rất nhiều lựa chọn, tùy thuộc mục đích sử dụng để chọn cho phù hợp. Ví dụ **MobileNet** có kích thước nhỏ hơn nhiều giúp tăng tốc độ mà vẫn đảm bảo performance chấp nhận được. **ResNet**, **DenseNet**... là những model tốt được sử dụng nhiều làm các backbone.

<img src="https://tryolabs.com/blog/images/blog/post-images/2018-01-18-faster-rcnn/vgg.b6e48b99.png" style="display:block; margin-left:auto; margin-right:auto">

*VGG16*

## Anchors
Vị trí object trong ảnh được xác định thông qua  bounding box. Ý tưởng ban đầu là xác định các tọa độ $x_{min}, y_{min}, x_{max}, y_{max}$ của bounding box. Tuy nhiên cách này gặp phải một số vấn đề:
- Phải đảm bảo $x_{min} < x_{max}$, $y_{min} < y_{max}$
- Các giá trị dự đoán có thể nằm ngoài bức ảnh

Do những vấn đề trên mà ý tưởng đó ít được sử dụng, từ đây anchor boxes ra đời. Anchor boxes là các boxes có kích thước và tỉ lệ khác nhau được đặt lên ảnh.

Ví dụ chúng ta có convolutional featura map với kích thước $conv_{width} \times conv_{height} \times conv_{depth}$, chúng ta sẽ tạo bộ anchors cho mỗi điểm trong $conv_{width} \times conv_{height}$. Một lưu ý quan trọng rằng mặc dù chúng ta định nghĩa anchors trên convolutional feature map nhưng anchors cuối cùng cần được tham chiếu đến ảnh ban đầu (chúng ta đã biết tỉ lệ ảnh ban đầu và feature map).

Giả sử ảnh ban đầu của chúng ta có kích thước $w \times h$, feature map có kích thước là $w/r \times h/r$ với $1/r$ là subsampling ratio. Nếu chúng ta định nghĩa 1 anchor box cho mỗi vị trí (spatial position) trên feature map thì ảnh ban đầu sẽ có một bộ anchor boxes, mỗi anchor box cách nhau $r$ pixel. Trong trường hợp của VGG, $r=16$ (224/14). Hiểu đơn giản thì tâm hai anchor box gần nhau trong feature map cách nhau 1 pixel thì trong ảnh gốc nó sẽ cách nhau $r$ pixels.

<img src="https://tryolabs.com/blog/images/blog/post-images/2018-01-18-faster-rcnn/anchors-centers.141181d6.png" style="display:block; margin-left:auto; margin-right:auto">

*Tâm của các anchors trong ảnh ban đầu*

Để xác định các bộ anchors người ta thường định nghĩa sẵn các kích thước (ví dụ 64px, 128px, 256px) và các tỉ lệ giữa width và height (ví dụ 0.5, 1, 1.5), cuối cùng sử dụng phương án kết hợp giữa chúng.

<img src="https://tryolabs.com/blog/images/blog/post-images/2018-01-18-faster-rcnn/anchors-progress.119e1e92.png" style="display:block; margin-left:auto; margin-right:auto">

*Anchors, Anchors cho 1 điểm, Anchors cho toàn bộ ảnh*

## Region Proposal Network
<img src="https://tryolabs.com/blog/images/blog/post-images/2018-01-18-faster-rcnn/rpn-architecture.99b6c089.png" style="display:block; margin-left:auto; margin-right:auto">

*RPN lấy feature map và tạo ta các region proposals*

Như đã đề cập RPN sẽ lấy tất cả các reference boxes (anchors) và đưa ra các region proposals tốt để phục vụ sau này. Để làm được điều này cùng xem RPN đã làm gì.
- RPN đưa ra probabitily mà anchor có chứa object. Chú ý rằng RPN không quan tâm class của object, nó chỉ quan tâm đó là object hay background thôi (2 nhóm).
- RPN đưa ra bounding box regression cho mỗi anchor để khớp tốt hơn với vị trí của ground-truth box (nếu )

RPN là fully convolutional network chứa toàn Conv layers. RPN nhận convolutional feature map từ pre-trained CNN model làm input. Sau đó cho qua Conv layer với 512 channels và kernel size là `3x3`. Cuối cùng chúng ta có 2 nhánh đều sử dụng Conv layer với kernel size `1x1`. Số channels lần lượt là $2k$ và $4k$.

<img src="https://tryolabs.com/blog/images/blog/post-images/2018-01-18-faster-rcnn/rpn-conv-layers.63c5bf86.png" style="display:block; margin-left:auto; margin-right:auto">

*Convolutional implementation của RPN, $k$ là số anchor boxes tại mỗi vị trí*

Đối với classification layer chúng ta có 2 dự đoán cho mỗi anchor: score mà anchor chứa object và score mà anchor chứa background (tổng 2 cái này bằng 1, thực chất có thể dùng logistic thay vì softmax). Do đó ở đây số channels đầu ra là $2k$.

Đối với regression để tinh chỉnh vị trí của anchor so với ground-truth box, mỗi anchor chúng ta sẽ có 4 offset values $t_{x}, t_{y}, t_{h}, t_{w}$ (chi tiết hơn xem thêm bài [R-CNN](https://huytranvan2010.github.io/R-CNN-Understanding/)). Do đó ở đây số channels đầu ra là $4k$.

### Training, target and loss function cho RPN
RPN thực hiện 2 dự đoán:
- Binary classfication
- Bounding box regression adjustment

Để thực hiện training cho RNP chúng ta lấy tất cả anchors và xếp chúng vào 2 loại:
- Foreground (có chứa object):
    - Anchors có IoU lớn nhất với ground-truth box, 
    - Anchors có $IoU > 0.7$ với **bất kỳ** ground truth box
- Background (không chứa object): anchors có $IoU < 0.3$ với ground truth box

Các anchors còn lại có IoU trong khoảng $(0.3, 0.7)$ không được sử dụng trong quá trình training.

Sau đó chúng ta sẽ lấy ngẫu nhiên các anchors để tạo mini-batch với kích thước 256 và cố gắng duy trì cân bằng giữa foreground và background. Điều này không phải lúc nào cũng thực hiện được, nó phụ thuộc vào objects và các anchors. Nếu số lượng *foreground anchor* không đủ sẽ lấy *background* anchor thay thế.

RPN sử dụng tất cả các anchors được chọn cho mini-batch để tính classification loss bằng cách sử dụng [binary cross entropy](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/). Tuy nhiên RPN chỉ sử dụng các anchors được gán nhãn là *foreground* để tính regression loss. Để tính targets for the regression ($t_{x}, t_{y}, t_{h}, t_{w}$) chúng ta sử dụng foreground anchor và groud-truth box gần nhất với nó.

>Giải thích thêm:
Công thức tính cross entropy:
$$H(y, \hat{y}) = \sum_i y_i \log \frac{1}{\hat{y}_i} = -\sum_i y_i \log \hat{y}_i$$
Ví dụ một anchor là foreground khi đó nó có $(y_0=1, y_1=0)$, đầu ta tính được $(\hat{y_0}=0.7, \hat{y_1}=0.3)$, thay vào công thức trên ta xác định được classification loss cho anchor box đó.
Nếu anchor box là foreground thì nó sẽ có các giá trị $t_{x}, t_{y}, t_{h}, t_{w}$ so với ground-truth box gần nhất (IoU max), đầu ta tính được $\hat{t}_{x}, \hat{t}_{y}, \hat{t}_{h}, \hat{t}_{w}$, khi đó ta cũng xác định được regression loss cho anchor đó.
Nếu anchor là background thì nó không có ground-truth box liên quan nên sẽ luôn để regression loss của nó bằng 0 mặc kê các giá trị tính ra.

Regression loss không sử dụng L1 hay L1 loss mà sử dụng smooth L1 loss.

### Post preprocessing RPN
**Non-max suppression (NMS)**

Các anchors thường chồng chập nhau, do đó sau RPN chúng ta nhận được rất nhiều proposals (từ anchors) chồng chập lên cùng object. Để giải quyết vấn đề này chúng ta áp dụng NMS. Sắp xếp các proposals theo chiều giảm của confidence score (chứa object). Duyệt lần lượt từ proposal đầu tiên, loại bỏ các proposal có IoU với proposal đầu tiên lớn hơn $threshold$. Làm tương tự với các proposals còn lại.

Cần chú ý đến **IoU threshold**:
- Nếu IoU threshold quá thấp (giữ lại ít proposals) chúng ta có thể bỏ sót các proposal thực sự chứa objects
- Nếu IoU threshold quá cao (giữ lại nhiều proposals) có thể dẫn đến có nhiều proposals cho cùng object
Giá trị IoU threshold thường được chọn là $0.6$.

**Proposal selection**

Sau khi thực hiện NMS chúng ta còn lại rất nhiều proposals nhưng chúng ta chỉ giữ lại top N proposals theo score. Trong bài báo gốc $N=2000$ được sử dụng, tuy nhiên có thể lấy $N$ thấp hơn ví dụ như $50$ và vẫn cho kết quả khá tốt.

**Standalone application**
Thực chất RPN có thể được sử dụng độc lập mà không cần the second stage model (phần classification và regression dựa trên Fast R-CNN). Trong bài toán chỉ có một loại object chúng ta có thể sử dụng probalitity tính được ở classification layer trong RPN làm class probability. Trong trường hợp này *foreground* là *class* và *background* là *not class*.

Một lợi ích nếu chỉ sử dụng RPN đó là tốc độ khi training và prediction vì RPN là mạng NN khá đơn giản với chỉ các Conv layers.

## RoI Pooling
Cơ chế của RoI Pooling mình đã trình bày khá đầy đủ trong bài viết về Fast R-CNN các bạn có thể xem thêm [tại đây](https://huytranvan2010.github.io/Fast-R-CNN-Understanding/).

<img src="https://tryolabs.com/blog/images/blog/post-images/2018-01-18-faster-rcnn/roi-architecture.7eaae6c2.png" style="display:block; margin-left:auto; margin-right:auto">

*RoI Pooling*

## Region-bases Convolutional Neural Network
Region-bases Convolutional Neural Network (R-CNN) là bước cuối cùng trong Faster R-CNN pipeline (chính xác làm làm tương tự như Fast R-CNN).

Phần feature map của region proposal đi qua RoI pooling và một số FC layers rồi tách thành 2 nhánh để:
- Phân loại proposal vào một trong các classes (tính cả background)
- Bounding box regression để tinh chỉnh lại vị trí của proposal

Trong bài báo gốc Faster R-CNN lấy feature map cho mỗi proposal, flatten ra và sử dụng 2 FC layers size 4096 với ReLu activation. Sau đó sử dụng 2 FC layers:
- FC layer có $N+1$ units, ở đây $N$ chính là tổng số classes, 1 tính đến background.
- FC layer với $4N$ units. Chúng ta muốn regression prediction $t_{x}, t_{y}, t_{h}, t_{w}$ cho mỗi possible class trong số $N$ classes.

<img src="https://tryolabs.com/blog/images/blog/post-images/2018-01-18-faster-rcnn/rcnn-architecture.6732b9bd.png" style="display:block; margin-left:auto; margin-right:auto">

*R-CNN architecture*

**Training and targets**
**Targets của R-CNN được tính gần giống trong RPN, tuy nhiên ở đây có tính đến các classes. Chúng ta lấy các proposals, ground-truth boxes để tính IoU giữa chúng.

Các proposals có $IoU > 0.5$ với bất kỳ ground-truth box nào được gán cho ground truth box đó. Proposals có IoU từ 0.1 đến 0.5 được coi là background. Không giống như RPN ở phần này không lấy proposal có $IoU=0$ làm background. Ở đây giả sử proposals của chúng ta là các proposals tốt có khả năng chứa vật thể nên chúng ta muốn nó học các trường hợp khó khó hơn (chứa một phần vật thể làm background). 

Targets cho bounding box regression $t_{x}, t_{y}, t_{h}, t_{w}$ được tính dựa vào proposals và ground-truth box tương ứng của nó (chắc với IoU lớn nhất).

Chúng ta cũng lấy ngẫu nhiên mini-batch với kích thước 64 trong đó có 25% là foreground proposals (with class) và 75% background.**

**Post processing**
Do mỗi bounding box trả về confidence scores cho các classes. Với bounding box có confidence score của background lớn nhất chúng ta sẽ loại bỏ. Còn các class khác sẽ lấy probability lớn nhất tương ứng với class.

Sau đó chúng ta sẽ áp dụng Non-max suppression để loại bỏ các bounding box chồng chập. Điều này được thực hiện thông qua việc gom lại các nhóm bounding boxes theo từng class và áp dụng trên đó.

## Training
Trong bài báo gốc Faster R-CNN được training bằng cách sử dụng multi-step approach, train các phần độc lập nhau sau đó merge (ghép lại) các trained weights trước khi thực hiện full training. Nhận thấy rằng việc huấn luyện end-to-end cho kết quả tốt hơn.

Sau khi ghép lại chungs ta có 4 losses (RPN và R-CNN mỗi cái có 2 losses). Chúng ta có traineable layers trong RNP và R-CNN, chúng ta có base network có thể được fine-tune hoặc không. Quyết định train base network phụ thuộc tương đồng dữ liệu giữa original dataset (ImageNet) và downstream dataset (object detection dataset). 

4 losses được kết hợp lại với nhau theo trọng số bởi vì chúng ta muốn classification classes có trọng số lớn hơn so với regression hay muốn R-CNN losses có trọng số lớn hơn so với RPN losses (NN sẽ phải chú tâm giảm loss của R-CNN nhiều hơn).

## Evaluation
Đối với bài tán object detection thường hay dùng mAP để đánh giá mô hình.

<img src="https://i1.wp.com/nttuan8.com/wp-content/uploads/2019/05/faster_rcnn_result.png?resize=768%2C439&ssl=1" style="display:block; margin-left:auto; margin-right:auto">


## Tài liệu tham khảo
1. https://arxiv.org/abs/1506.01497
2. https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/
3. https://www.coursera.org/lecture/deep-learning-in-computer-vision/faster-r-cnn-DM0hz
4. https://blog.paperspace.com/faster-r-cnn-explained-object-detection/
5. https://towardsdatascience.com/understanding-fast-r-cnn-and-faster-r-cnn-for-object-detection-adbb55653d97
6. https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8
7. https://towardsdatascience.com/faster-rcnn-1506-01497-5c8991b0b6d3
8. https://towardsdatascience.com/understanding-region-of-interest-part-1-roi-pooling-e4f5dd65bb44
9. https://github.com/rafaelpadilla/Object-Detection-Metrics
10. https://stats.stackexchange.com/questions/351874/how-to-interpret-smooth-l1-loss
11. https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a
12. https://blog.zenggyu.com/en/post/2018-12-16/an-introduction-to-evaluation-metrics-for-object-detection/
13. https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c 
14. https://manalelaidouni.github.io/Evaluating-Object-Detection-Models-Guide-to-Performance-Metrics.html#precision-x-recall-curve
15. https://www.programmersought.com/article/97705419705/
16. https://towardsdatascience.com/faster-rcnn-1506-01497-5c8991b0b6d3
17. https://github.com/huytranvan2010/Faster-R-CNN-Understanding

