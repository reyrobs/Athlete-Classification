<div align="center">
<div id="top"></div>
<h2 align="center">Athlete Classification with Open CV</h3>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-description">Project Description</a>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">References</a></li>
  </ol>
</details>

## Project Description

The purpose of this project is to perform classification of famous athletes with the use of OpenCV. We have made use of two seperate datasets, both of which contained 5 athletes to be classified. The first one was supplied from Dhaval Patel [[1]](#1), from whom we also got the inspiration to do to the project. The second dataset was obtained with the help of web scraping. The classifiers used to perform the classification were Support Vector Machines, Random Forrest and Logistic Regression, all with the help of GridSearchCV, such as to be able to find the best hyper parameters. 
</br>
</br>
The first step after obtaining the data was to crop the face of each athlete and use those cropped images for the training of our classifiers, since the face is the main part used to differentiate the athletes from each other. In order to detect the faces, we have made use of the OpenCV Haar Cascades xml files, which act themselves as a classifier. Once the faces cropped, we use wavelet transform to create a new image from the cropped image in order to highlight the essential features of the face. Since the original image also contains important features, we stack both the original image and the one obtained from applying the wavelet transform and feed it as input to the classifiers. 
</br>
</br>
Our results show that we obtained much better results with the first dataset, i.e the one already provided to us. We believe that the reason for this has to do with the quality between the two datasets. The first dataset simply had better resolution, which resulted in more useful information being conveyed in the image. 

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- Data Exploration -->
## Data Exploration 

### Dataset 1 Histogram 
![alt text](https://github.com/reyrobs/Athlete-Classification/blob/main/dataset1_histo.png?raw=true)
### Dataset 2 Histogram 
![alt text](https://github.com/reyrobs/Athlete-Classification/blob/main/dataset2_histo.png?raw=true)

## Face detection 

![alt text](https://github.com/reyrobs/Athlete-Classification/blob/main/sharapova_resized.png?raw=true)
&emsp;&emsp;&emsp;&emsp;
![alt text](https://github.com/reyrobs/Athlete-Classification/blob/main/sharapova_face_resized.png?raw=true)
&emsp;&emsp;&emsp;&emsp;
![alt text](https://github.com/reyrobs/Athlete-Classification/blob/main/sharapova_cropped_resized.png?raw=true)
<br>
![alt text](https://github.com/reyrobs/Athlete-Classification/blob/main/andy_murray_resized.png?raw=true)
&emsp;&emsp;&emsp;&emsp;
![alt text](https://github.com/reyrobs/Athlete-Classification/blob/main/andy_murray_face_resized.png?raw=true)
&emsp;&emsp;&emsp;&emsp;
![alt text](https://github.com/reyrobs/Athlete-Classification/blob/main/andy_murray_cropped_resized.png?raw=true)



<p align="right">(<a href="#top">back to top</a>)</p>

## Wavelet Transform
Once the images were cropped, we made use of the wavelet transform in order to extract the important features of the face. The main idea of the wavelet transform is the ability to capture both local spectral and temporal information. More information about it can be found here [[2]](#2).

![alt text](https://github.com/reyrobs/Athlete-Classification/blob/main/sharapova_cropped_resized.png?raw=true)
&emsp;&emsp;&emsp;&emsp;
![alt text](https://github.com/reyrobs/Athlete-Classification/blob/main/sharapova_wavelet_resized.png?raw=true)
<br>
![alt text](https://github.com/reyrobs/Athlete-Classification/blob/main/andy_murray_cropped_resized.png?raw=true)
&emsp;&emsp;&emsp;&emsp;
![alt text](https://github.com/reyrobs/Athlete-Classification/blob/main/murray_wavelet_resized.png?raw=true)

<p align="right">(<a href="#top">back to top</a>)</p>

## Results and discussion

### Dataset 1

<table>
<thead>
  <tr>
    <th>Classifier</th>
    <th>Testing Accuracy</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>SVM</td>
    <td>0.914</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>0.723</td>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>0.894</td>
  </tr>
</tbody>
</table>

### Dataset 2

<table>
<thead>
  <tr>
    <th>Classifier</th>
    <th>Testing Accuracy</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>SVM</td>
    <td>0.545</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>0.455</td>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>0.515</td>
  </tr>
</tbody>
</table>

From the results obtained above, we can see that SVM performed the best for both datasets. We clearly see a much better performance on the first dataset however, and we believe this is due to the better resolution of the images used, which allowed for more features to be captured. Additionally, we made use of GridSearchCV in order to try and find the best hyperparameters for our models. 

<p align="right">(<a href="#top">back to top</a>)</p>

## Confusion matrix for best model
![alt text](https://github.com/reyrobs/Athlete-Classification/blob/main/cm.png?raw=true)

<!-- CONTACT -->
## Contact

Robert Rey - [LinkedIn](https://www.linkedin.com/in/robert-rey-36689a103/)

Project Link: [Athlete Classification](https://github.com/reyrobs/Athlete-Classification)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>




## References
<a id="1">[1]</a> 
Dhaval Patel,
https://codebasics.io/

<a id="2">[2]</a> 
The Wavelet Transform,
Shawhin Talebi,
https://towardsdatascience.com/the-wavelet-transform-e9cfa85d7b34

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
