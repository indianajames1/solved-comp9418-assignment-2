Download Link: https://assignmentchef.com/product/solved-comp9418-assignment-2
<br>
<span style="font-size: 2.61792em; letter-spacing: -1px;">1             [50 Marks] Expectation Maximisation</span>

Consider a model with continuous observed variables <strong>x </strong>∈ R<em><sup>D </sup></em>and hidden variables <strong>t </strong>∈ {0<em>,</em>1}<em><sup>K </sup></em>and <strong>z </strong>∈ R<em><sup>Q</sup></em>. The hidden variable <strong>t </strong>is a <em>K</em>-dimensional binary random variable with a 1-of-<em>K </em>representation, where <em>t<sub>k </sub></em>∈ {0<em>,</em>1} and <sup>P</sup><em><sub>k </sub>t<sub>k </sub></em>= 1, i.e. exactly one component of <em>t<sub>k </sub></em>is equal to 1 while all others are equal to 0. The prior distribution over <strong>t </strong>is given by

<em>p</em>(<em>t<sub>k </sub></em>= 1|<em>θ</em>) = <em>π<sub>k</sub>,                                                              </em>(1)

where mixing weights  satisfy 0 ≤ <em>π<sub>k </sub></em>≤ 1 and                 = 1. This can also be

<table width="643">

 <tbody>

  <tr>

   <td width="266">written in the form</td>

   <td width="173"> </td>

   <td width="203"> </td>

  </tr>

  <tr>

   <td width="266"> </td>

   <td width="173"><em>K p</em>(<strong>t</strong>|<em>θ</em>) = <sup>Y</sup><em>π<sub>k</sub><sup>t</sup></em><em><sup>k</sup>.</em></td>

   <td width="203">(2)</td>

  </tr>

 </tbody>

</table>

<em>k</em>=1

COMP9418, UNSW Sydney                       Advanced Topics in Statistical Machine Learning, 18s2

Hidden variable <strong>z </strong>is a <em>Q</em>-dimensional continuous random variable with prior distribution

<em>p</em>(<strong>z</strong>|<em>θ</em>) = <em>p</em>(<strong>z</strong>) = N(<strong>0</strong><em>,</em><strong>I</strong>)<em>.                                                         </em>(3)

The conditional likelihood of <strong>x </strong>given <strong>z </strong>and <em>t<sub>k </sub></em>= 1 is a Gaussian defined as

<em>p</em>(<strong>x</strong>|<strong>z</strong><em>,t<sub>k </sub></em>= 1<em>,</em><em>θ</em>) = N(<strong>x</strong>|<strong>W</strong><em><sub>k</sub></em><strong>z </strong>+ <strong>b</strong><em><sub>k</sub>,</em><strong>Ψ</strong>)<em>,                                               </em>(4)

where <strong>W</strong><em><sub>k </sub></em>∈ R<em><sup>D</sup></em><sup>×<em>Q</em></sup>, <strong>b</strong><em><sub>k </sub></em>∈ R<em><sup>D </sup></em>and <strong>Ψ </strong>∈ R<em><sup>D</sup></em><sup>×<em>D </em></sup>is a <em>diagonal </em>covariance matrix. Another way to express this is

<em>K</em>

<em>p</em>(<strong>x</strong>|<strong>z</strong><em>,</em><strong>t</strong><em>,</em><em>θ</em>) = <sup>Y</sup>N(<strong>x</strong>|<strong>W</strong><em><sub>k</sub></em><strong>z </strong>+ <strong>b</strong><em><sub>k</sub>,</em><strong>Ψ</strong>)<em><sup>t</sup></em><em><sup>k</sup>.                                                 </em>(5)

<em>k</em>=1

Let us collectively denote the set of all observed variables by <strong>X </strong> and hidden variables by <strong>Z </strong> and <strong>T </strong>. The joint distribution is denoted by <em>p</em>(<strong>Z</strong><em>,</em><strong>T</strong><em>,</em><strong>X</strong>|<em>θ</em>), and is governed by the set of model parameters .

In the questions below, unless otherwise stated explicitly, you must <strong>show all your working</strong>. Omission of details or derivations may yield a reduction in the corresponding marks.

<ol>

 <li>[5 marks] Draw the graphical representation for this probabilistic model, making sure to include the parameters <em>θ </em>in the graph. (Non-random variables can be included similarly to random variables, except that circles are not drawn around them).</li>

 <li>[5 marks] In terms of <em>K,D,Q</em>, give an expression for the number of parameters we are required to estimate under this model.</li>

 <li>[10 marks] In the E-step of the expectation maximization (em) algorithm, we are required to compute the expected sufficient statistics of the posterior over hidden variables. The posterior responsibility of mixture component <em>k </em>for a data-point <em>n </em>is expressed as</li>

</ol>

def                                                     old

<em>r</em><em>nk </em>= <em>p</em>(<em>t</em><em>nk </em>= 1|<strong>x</strong><em>n,</em><em>θ </em>) = E<em>p</em>(<em>t</em><em>nk</em>|<strong>x</strong><em>,</em><em>θ</em>old)[<em>t</em><em>nk</em>]<em>.                                         </em>(6)

The conditional posterior over local hidden factor <strong>z</strong><em><sub>n </sub></em>is a Gaussian with mean <strong>m</strong><em><sub>nk </sub></em>and covariance <strong>C</strong><em><sub>nk</sub></em>,

<em>p</em>(<strong>z</strong><em><sub>n</sub></em>|<em>t<sub>nk </sub></em>= 1<em>,</em><strong>x</strong><em><sub>n</sub>,</em><em>θ</em><sup>old</sup>) = N(<strong>z</strong><em><sub>n</sub></em>|<strong>m</strong><em><sub>nk</sub>,</em><strong>C</strong><em><sub>nk</sub></em>)<em>.                                         </em>(7)

The covariance is given by

<strong>C</strong><em>,                                                    </em>(8)

where

def

<strong>m</strong><em>nk </em>= E<em>p</em><sub>(<strong>z</strong></sub><em><sub>n</sub></em>|<em><sub>t</sub></em><em><sub>nk</sub></em>=1<em><sub>,</sub></em><strong><sub>x</sub></strong><em><sub>n</sub></em><em><sub>,</sub></em><em><sub>θ</sub></em>old<sub>)</sub>[<strong>z</strong><em>n</em>]<em>, </em>and <strong>S</strong><em>. </em>(9) i) [5 marks] Give analytical expressions for the responsibilities <em>r<sub>nk </sub></em>and the expected sufficient statistics <strong>m</strong><em><sub>nk </sub></em>and <strong>S</strong><em><sub>nk </sub></em>in terms of the old model parameters <em>θ</em><sup>old</sup>.

<ol>

 <li>[1 marks] To de-clutter notation and simplify subsequent analysis, it is helpful to introduce <em>augmented </em>factor loading matrix and hidden factor vector,</li>

</ol>

<strong>W</strong><sup>˜</sup><em> ,         </em>and        <strong>z</strong>˜ <em>.                          </em>(10)

Accordingly, give expressions for the sufficient statistics of the conditional posterior on augmented hidden factor vectors,

def E <em>n nk n </em>old ˜ <strong>m</strong>˜ <em>nk </em>= <em>p</em>(<strong>z</strong>˜ |<em>t </em>=1<em>,</em><strong>x </strong><em>,</em><em>θ </em>)[<strong>z</strong>˜<em>n</em>]<em>, </em>and <strong>S</strong>

Note you need only express this in terms of <strong>m</strong><em><sub>nk </sub></em>and <strong>S</strong><em><sub>nk</sub></em>.

COMP9418, UNSW Sydney                       Advanced Topics in Statistical Machine Learning, 18s2

<ul>

 <li>[4 marks] Show that the sufficient statistics of the joint posterior factorise as follows,</li>

</ul>

<em>.</em>

<ol>

 <li>[10 marks] Write down the full expression for the <em>expected complete-data log likelihood </em>(also known as <em>auxiliary function</em>) for this model,</li>

</ol>

<em>Q</em>(<em>θ,</em><em>θ</em><sup>old</sup>) <sup>def</sup>= E<em>p</em><sub>(<strong>Z</strong><em>,</em><strong>T</strong>|<strong>X</strong><em>,</em></sub><em><sub>θ</sub></em>old<sub>)</sub>[log<em>p</em>(<strong>Z</strong><em>,</em><strong>T</strong><em>,</em><strong>X</strong>|<em>θ</em>)]<em>. </em>(12) e) [20 marks] Optimize the auxiliary function <em>Q </em>w.r.t. model parameters <em>θ </em>to obtain M-step updates. Show all your working and highlight each individual update equation.

<h1>2             [50 Marks] Practical Part</h1>

See Jupyter notebook comp9418 ass2.ipynb.