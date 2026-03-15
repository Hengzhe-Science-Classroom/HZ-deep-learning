// === Chapter 0: Linear Algebra Review ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch00',
    number: 0,
    title: 'Linear Algebra Review',
    subtitle: 'Vectors, matrices, decompositions, and the geometric language of deep learning',
    sections: [
        // ========== SECTION 1: Vectors & Matrices ==========
        {
            id: 'sec01-vectors-matrices',
            title: 'Vectors & Matrices',
            content: `
<h2>0.1 Vectors & Matrices</h2>

<div class="env-block intuition">
<strong>Why Linear Algebra?</strong> Deep learning is, at its core, a sequence of linear transformations interleaved with nonlinearities. Every forward pass through a neural network multiplies input vectors by weight matrices, adds biases, and applies activation functions. Understanding how matrices transform space is the single most important mathematical prerequisite for deep learning.
</div>

<h3>Scalars, Vectors, and Matrices</h3>

<p>We begin with the basic objects of linear algebra. A <em>scalar</em> is a single number, such as \\(x \\in \\mathbb{R}\\). A <em>vector</em> is an ordered list of numbers:</p>

\\[
\\mathbf{x} = \\begin{bmatrix} x_1 \\\\ x_2 \\\\ \\vdots \\\\ x_n \\end{bmatrix} \\in \\mathbb{R}^n.
\\]

<p>A <em>matrix</em> is a rectangular array of numbers:</p>

\\[
\\mathbf{A} = \\begin{bmatrix} a_{11} & a_{12} & \\cdots & a_{1n} \\\\ a_{21} & a_{22} & \\cdots & a_{2n} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\ a_{m1} & a_{m2} & \\cdots & a_{mn} \\end{bmatrix} \\in \\mathbb{R}^{m \\times n}.
\\]

<div class="env-block definition">
<strong>Definition 0.1.1 (Tensor).</strong> A <em>tensor</em> is the generalization of scalars, vectors, and matrices to arbitrary dimensions. A scalar is a 0-dimensional tensor, a vector is 1-dimensional, a matrix is 2-dimensional, and a 3D array (e.g., an RGB image with height, width, and channels) is a 3-dimensional tensor. In deep learning frameworks such as PyTorch and TensorFlow, all data is stored as tensors.
</div>

<h3>Vector Operations</h3>

<p>Two fundamental operations on vectors are <em>addition</em> and <em>scalar multiplication</em>. For \\(\\mathbf{x}, \\mathbf{y} \\in \\mathbb{R}^n\\) and \\(\\alpha \\in \\mathbb{R}\\):</p>

\\[
\\mathbf{x} + \\mathbf{y} = \\begin{bmatrix} x_1 + y_1 \\\\ \\vdots \\\\ x_n + y_n \\end{bmatrix}, \\qquad \\alpha \\mathbf{x} = \\begin{bmatrix} \\alpha x_1 \\\\ \\vdots \\\\ \\alpha x_n \\end{bmatrix}.
\\]

<div class="env-block definition">
<strong>Definition 0.1.2 (Dot Product).</strong> The <em>dot product</em> (or <em>inner product</em>) of two vectors \\(\\mathbf{x}, \\mathbf{y} \\in \\mathbb{R}^n\\) is
\\[
\\mathbf{x} \\cdot \\mathbf{y} = \\mathbf{x}^\\top \\mathbf{y} = \\sum_{i=1}^n x_i y_i.
\\]
The dot product is a scalar that measures the "alignment" between two vectors. If \\(\\mathbf{x} \\cdot \\mathbf{y} = 0\\), the vectors are <em>orthogonal</em> (perpendicular).
</div>

<div class="env-block remark">
<strong>Geometric Interpretation.</strong> The dot product can also be written as \\(\\mathbf{x} \\cdot \\mathbf{y} = \\|\\mathbf{x}\\| \\, \\|\\mathbf{y}\\| \\cos\\theta\\), where \\(\\theta\\) is the angle between the two vectors. This formula is the workhorse behind attention mechanisms in transformers, where the dot product between query and key vectors measures their similarity.
</div>

<h3>Matrix-Vector Multiplication</h3>

<p>Given \\(\\mathbf{A} \\in \\mathbb{R}^{m \\times n}\\) and \\(\\mathbf{x} \\in \\mathbb{R}^n\\), the product \\(\\mathbf{y} = \\mathbf{A}\\mathbf{x} \\in \\mathbb{R}^m\\) is defined by</p>

\\[
y_i = \\sum_{j=1}^n a_{ij} x_j, \\quad i = 1, \\ldots, m.
\\]

<div class="env-block intuition">
<strong>Matrices as Transformations.</strong> A matrix \\(\\mathbf{A} \\in \\mathbb{R}^{2 \\times 2}\\) defines a linear transformation of the plane. Every point \\(\\mathbf{x}\\) is mapped to \\(\\mathbf{A}\\mathbf{x}\\). The columns of \\(\\mathbf{A}\\) tell you where the standard basis vectors \\(\\mathbf{e}_1\\) and \\(\\mathbf{e}_2\\) land. The unit square spanned by these basis vectors gets mapped to a parallelogram spanned by the columns of \\(\\mathbf{A}\\). This geometric view is central to understanding how neural network layers reshape data.
</div>

<div class="viz-placeholder" data-viz="viz-matrix-transform"></div>

<h3>Matrix-Matrix Multiplication</h3>

<div class="env-block definition">
<strong>Definition 0.1.3 (Matrix Multiplication).</strong> For \\(\\mathbf{A} \\in \\mathbb{R}^{m \\times p}\\) and \\(\\mathbf{B} \\in \\mathbb{R}^{p \\times n}\\), the product \\(\\mathbf{C} = \\mathbf{A}\\mathbf{B} \\in \\mathbb{R}^{m \\times n}\\) is given by
\\[
c_{ij} = \\sum_{k=1}^p a_{ik} b_{kj}.
\\]
Matrix multiplication is <em>associative</em> and <em>distributive</em>, but <strong>not commutative</strong>: in general, \\(\\mathbf{A}\\mathbf{B} \\neq \\mathbf{B}\\mathbf{A}\\).
</div>

<div class="env-block remark">
<strong>Composition of Transformations.</strong> Matrix multiplication corresponds to the composition of linear transformations. If \\(\\mathbf{A}\\) rotates and \\(\\mathbf{B}\\) scales, then \\(\\mathbf{A}\\mathbf{B}\\) first scales (applies \\(\\mathbf{B}\\)) then rotates (applies \\(\\mathbf{A}\\)). In a neural network, stacking layers without activation functions is equivalent to multiplying weight matrices, so the entire network collapses to a single linear map. This is precisely why nonlinear activations are essential.
</div>

<div class="env-block example">
<strong>Example 0.1.4 (A Neural Network Layer).</strong> A single fully connected layer computes
\\[
\\mathbf{h} = \\sigma(\\mathbf{W}\\mathbf{x} + \\mathbf{b}),
\\]
where \\(\\mathbf{W} \\in \\mathbb{R}^{d_{\\text{out}} \\times d_{\\text{in}}}\\) is the weight matrix, \\(\\mathbf{b} \\in \\mathbb{R}^{d_{\\text{out}}}\\) is the bias vector, and \\(\\sigma\\) is a nonlinear activation. The linear part \\(\\mathbf{W}\\mathbf{x} + \\mathbf{b}\\) is an <em>affine transformation</em>: it stretches, rotates, and shifts the input space.
</div>
`,
            visualizations: [
                {
                    id: 'viz-matrix-transform',
                    title: 'Interactive Matrix Transformation',
                    description: 'Adjust the sliders to change the 2\\(\\times\\)2 matrix entries. Watch how the unit square (blue outline) transforms into a parallelogram (teal filled). The determinant tells you the signed area scaling factor.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 50 });
                        let a = 1, b = 0, c = 0, d = 1;

                        const sA = VizEngine.createSlider(controls, 'a', -3, 3, 1, 0.1, v => { a = v; });
                        const sB = VizEngine.createSlider(controls, 'b', -3, 3, 0, 0.1, v => { b = v; });
                        const sC = VizEngine.createSlider(controls, 'c', -3, 3, 0, 0.1, v => { c = v; });
                        const sD = VizEngine.createSlider(controls, 'd', -3, 3, 1, 0.1, v => { d = v; });

                        // Preset buttons
                        const presetDiv = document.createElement('div');
                        presetDiv.style.cssText = 'display:flex;gap:4px;flex-wrap:wrap;margin-top:4px;';
                        const presets = [
                            { label: 'Rotation 45\u00b0', vals: [0.707, -0.707, 0.707, 0.707] },
                            { label: 'Scale 2x', vals: [2, 0, 0, 2] },
                            { label: 'Shear', vals: [1, 1, 0, 1] },
                            { label: 'Reflection', vals: [-1, 0, 0, 1] },
                            { label: 'Identity', vals: [1, 0, 0, 1] }
                        ];
                        presets.forEach(p => {
                            VizEngine.createButton(presetDiv, p.label, () => {
                                [a, b, c, d] = p.vals;
                                sA.value = a; sA.dispatchEvent(new Event('input'));
                                sB.value = b; sB.dispatchEvent(new Event('input'));
                                sC.value = c; sC.dispatchEvent(new Event('input'));
                                sD.value = d; sD.dispatchEvent(new Event('input'));
                            });
                        });
                        controls.appendChild(presetDiv);

                        function draw() {
                            viz.clear();
                            viz.drawGrid();
                            viz.drawAxes();

                            // Original unit square
                            viz.drawPolygon(
                                [[0,0],[1,0],[1,1],[0,1]],
                                null, viz.colors.blue + '88', 1.5
                            );

                            // Transformed parallelogram
                            const M = [[a, b], [c, d]];
                            viz.drawTransformedUnitSquare(M, viz.colors.teal + '22', viz.colors.teal, 2);

                            // Basis vectors (original)
                            viz.drawVector(0, 0, 1, 0, viz.colors.blue + '66', '', 1.5);
                            viz.drawVector(0, 0, 0, 1, viz.colors.blue + '66', '', 1.5);

                            // Transformed basis vectors
                            viz.drawVector(0, 0, a, c, viz.colors.orange, 'Ae\u2081', 2.5);
                            viz.drawVector(0, 0, b, d, viz.colors.green, 'Ae\u2082', 2.5);

                            // Info text
                            const det = VizEngine.det2(M);
                            viz.screenText(
                                'A = [' + a.toFixed(1) + ', ' + b.toFixed(1) + '; ' + c.toFixed(1) + ', ' + d.toFixed(1) + ']    det(A) = ' + det.toFixed(2),
                                viz.width / 2, 22, viz.colors.white, 13
                            );

                            if (Math.abs(det) < 0.01) {
                                viz.screenText('Singular! (det \u2248 0: columns are linearly dependent)', viz.width / 2, viz.height - 18, viz.colors.red, 12);
                            }
                        }

                        viz.animate(draw);
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Let \\(\\mathbf{A} = \\begin{bmatrix} 2 & 1 \\\\ 0 & 3 \\end{bmatrix}\\) and \\(\\mathbf{x} = \\begin{bmatrix} 1 \\\\ -1 \\end{bmatrix}\\). Compute \\(\\mathbf{A}\\mathbf{x}\\) and describe geometrically what \\(\\mathbf{A}\\) does to the standard basis vectors.',
                    hint: 'The first column of \\(\\mathbf{A}\\) is where \\(\\mathbf{e}_1 = (1,0)^\\top\\) lands; the second column is where \\(\\mathbf{e}_2 = (0,1)^\\top\\) lands.',
                    solution: '\\(\\mathbf{A}\\mathbf{x} = \\begin{bmatrix} 2(1) + 1(-1) \\\\ 0(1) + 3(-1) \\end{bmatrix} = \\begin{bmatrix} 1 \\\\ -3 \\end{bmatrix}\\). Geometrically, \\(\\mathbf{A}\\) maps \\(\\mathbf{e}_1\\) to \\((2, 0)^\\top\\) (stretches the x-axis by 2) and \\(\\mathbf{e}_2\\) to \\((1, 3)^\\top\\) (stretches in y by 3 and shears in x by 1). It is an upper-triangular transformation combining scaling and shearing.'
                },
                {
                    question: 'Explain why \\(\\mathbf{A}\\mathbf{B} \\neq \\mathbf{B}\\mathbf{A}\\) in general, and give a concrete 2\\(\\times\\)2 example where the products differ.',
                    hint: 'Try a rotation matrix and a scaling matrix. Rotating then scaling is different from scaling then rotating (unless the scaling is uniform).',
                    solution: 'Let \\(\\mathbf{A} = \\begin{bmatrix} 0 & -1 \\\\ 1 & 0 \\end{bmatrix}\\) (90\\(^\\circ\\) rotation) and \\(\\mathbf{B} = \\begin{bmatrix} 2 & 0 \\\\ 0 & 1 \\end{bmatrix}\\) (scale x by 2). Then \\(\\mathbf{A}\\mathbf{B} = \\begin{bmatrix} 0 & -1 \\\\ 2 & 0 \\end{bmatrix}\\) but \\(\\mathbf{B}\\mathbf{A} = \\begin{bmatrix} 0 & -2 \\\\ 1 & 0 \\end{bmatrix}\\). They differ because composition of transformations depends on order: rotating then scaling produces a different result than scaling then rotating when the scaling is anisotropic.'
                },
                {
                    question: 'A two-layer neural network without activation functions computes \\(\\mathbf{y} = \\mathbf{W}_2(\\mathbf{W}_1 \\mathbf{x} + \\mathbf{b}_1) + \\mathbf{b}_2\\). Show that this is equivalent to a single affine transformation \\(\\mathbf{y} = \\mathbf{W}\\mathbf{x} + \\mathbf{b}\\). What does this imply about the necessity of nonlinear activations?',
                    hint: 'Expand the expression and collect terms involving \\(\\mathbf{x}\\) and constant terms.',
                    solution: 'Expanding: \\(\\mathbf{y} = \\mathbf{W}_2 \\mathbf{W}_1 \\mathbf{x} + \\mathbf{W}_2 \\mathbf{b}_1 + \\mathbf{b}_2\\). Setting \\(\\mathbf{W} = \\mathbf{W}_2 \\mathbf{W}_1\\) and \\(\\mathbf{b} = \\mathbf{W}_2 \\mathbf{b}_1 + \\mathbf{b}_2\\), we get \\(\\mathbf{y} = \\mathbf{W}\\mathbf{x} + \\mathbf{b}\\), a single affine map. This means stacking linear layers without activations gains no representational power over a single layer. Nonlinear activations (ReLU, sigmoid, etc.) break this composability and allow the network to represent nonlinear decision boundaries.'
                }
            ]
        },

        // ========== SECTION 2: Norms & Inner Products ==========
        {
            id: 'sec02-norms-inner-products',
            title: 'Norms & Inner Products',
            content: `
<h2>0.2 Norms & Inner Products</h2>

<p>To do anything useful with vectors, we need to measure their <em>size</em> and the <em>angle</em> between them. Norms provide the notion of size; inner products provide both size and angle. These concepts appear everywhere in deep learning: loss functions, regularization penalties, gradient norms, and attention scores all rely on norms and inner products.</p>

<h3>Vector Norms</h3>

<div class="env-block definition">
<strong>Definition 0.2.1 (\\(L^p\\) Norm).</strong> For \\(\\mathbf{x} \\in \\mathbb{R}^n\\) and \\(p \\geq 1\\), the \\(L^p\\) norm is
\\[
\\|\\mathbf{x}\\|_p = \\left( \\sum_{i=1}^n |x_i|^p \\right)^{1/p}.
\\]
Special cases:
<ul>
<li>\\(p = 1\\): the <em>Manhattan norm</em> \\(\\|\\mathbf{x}\\|_1 = \\sum_i |x_i|\\). Used in L1 regularization (Lasso) to promote sparsity.</li>
<li>\\(p = 2\\): the <em>Euclidean norm</em> \\(\\|\\mathbf{x}\\|_2 = \\sqrt{\\sum_i x_i^2}\\). The most common norm in deep learning; used in L2 regularization (weight decay).</li>
<li>\\(p = \\infty\\): the <em>max norm</em> \\(\\|\\mathbf{x}\\|_\\infty = \\max_i |x_i|\\). Used in max-norm constraints for embedding layers.</li>
</ul>
</div>

<div class="env-block intuition">
<strong>The Unit Ball.</strong> The <em>unit ball</em> \\(\\{\\mathbf{x} : \\|\\mathbf{x}\\|_p \\leq 1\\}\\) has a characteristic shape for each \\(p\\). When \\(p = 2\\), it is the familiar circle (or sphere in higher dimensions). When \\(p = 1\\), it is a diamond (or cross-polytope). As \\(p \\to \\infty\\), the ball approaches a square (or hypercube). This geometry matters for regularization: L1 regularization's diamond-shaped constraint set has corners on the axes, which is why it drives parameters exactly to zero (sparsity). L2 regularization's circular constraint set shrinks parameters uniformly without inducing exact zeros.
</div>

<div class="viz-placeholder" data-viz="viz-norm-ball"></div>

<h3>Matrix Norms</h3>

<div class="env-block definition">
<strong>Definition 0.2.2 (Frobenius Norm).</strong> For \\(\\mathbf{A} \\in \\mathbb{R}^{m \\times n}\\), the Frobenius norm is
\\[
\\|\\mathbf{A}\\|_F = \\sqrt{\\sum_{i=1}^m \\sum_{j=1}^n a_{ij}^2} = \\sqrt{\\operatorname{tr}(\\mathbf{A}^\\top \\mathbf{A})}.
\\]
This is the "flattened \\(L^2\\) norm," treating the matrix as a long vector of \\(mn\\) entries.
</div>

<div class="env-block definition">
<strong>Definition 0.2.3 (Spectral Norm).</strong> The spectral norm of \\(\\mathbf{A}\\) is
\\[
\\|\\mathbf{A}\\|_2 = \\sigma_{\\max}(\\mathbf{A}),
\\]
the largest singular value of \\(\\mathbf{A}\\). It measures the maximum factor by which \\(\\mathbf{A}\\) can stretch any unit vector. Spectral normalization, a technique used in GANs, constrains weight matrices to have \\(\\|\\mathbf{W}\\|_2 = 1\\).
</div>

<h3>Inner Products and Orthogonality</h3>

<div class="env-block definition">
<strong>Definition 0.2.4 (Orthogonality).</strong> Two vectors \\(\\mathbf{x}, \\mathbf{y} \\in \\mathbb{R}^n\\) are <em>orthogonal</em> if \\(\\mathbf{x}^\\top \\mathbf{y} = 0\\). A set of vectors \\(\\{\\mathbf{v}_1, \\ldots, \\mathbf{v}_k\\}\\) is <em>orthonormal</em> if
\\[
\\mathbf{v}_i^\\top \\mathbf{v}_j = \\delta_{ij} = \\begin{cases} 1 & i = j, \\\\ 0 & i \\neq j. \\end{cases}
\\]
</div>

<div class="env-block theorem">
<strong>Theorem 0.2.5 (Cauchy-Schwarz Inequality).</strong> For any \\(\\mathbf{x}, \\mathbf{y} \\in \\mathbb{R}^n\\),
\\[
|\\mathbf{x}^\\top \\mathbf{y}| \\leq \\|\\mathbf{x}\\|_2 \\, \\|\\mathbf{y}\\|_2,
\\]
with equality if and only if \\(\\mathbf{x}\\) and \\(\\mathbf{y}\\) are linearly dependent. This inequality guarantees that the cosine similarity \\(\\cos \\theta = \\frac{\\mathbf{x}^\\top \\mathbf{y}}{\\|\\mathbf{x}\\|_2 \\|\\mathbf{y}\\|_2}\\) always lies in \\([-1, 1]\\).
</div>

<div class="env-block example">
<strong>Example 0.2.6 (Cosine Similarity in NLP).</strong> Word embeddings map words to vectors in \\(\\mathbb{R}^d\\). The <em>cosine similarity</em>
\\[
\\text{sim}(\\mathbf{u}, \\mathbf{v}) = \\frac{\\mathbf{u}^\\top \\mathbf{v}}{\\|\\mathbf{u}\\|_2 \\|\\mathbf{v}\\|_2}
\\]
measures semantic similarity. In Word2Vec, the famous analogy "king - man + woman \\(\\approx\\) queen" works because these vector arithmetic operations preserve cosine similarity structure.
</div>
`,
            visualizations: [
                {
                    id: 'viz-norm-ball',
                    title: '\\(L^p\\) Norm Unit Ball',
                    description: 'Slide the parameter \\(p\\) to see how the shape of the unit ball \\(\\{\\mathbf{x} : \\|\\mathbf{x}\\|_p = 1\\}\\) changes. At \\(p=1\\) (diamond) corners sit on axes, promoting sparsity. At \\(p=2\\) (circle) the ball is rotationally symmetric. As \\(p \\to \\infty\\) it approaches a square.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 100 });
                        let p = 2;
                        VizEngine.createSlider(controls, 'p', 0.5, 10, 2, 0.1, v => { p = v; });

                        function draw() {
                            viz.clear();
                            viz.drawGrid();
                            viz.drawAxes();

                            const ctx = viz.ctx;
                            const N = 360;

                            // Draw unit ball boundary
                            ctx.beginPath();
                            for (let i = 0; i <= N; i++) {
                                const theta = (2 * Math.PI * i) / N;
                                const cosT = Math.cos(theta);
                                const sinT = Math.sin(theta);
                                // For |x|^p + |y|^p = 1, parameterize:
                                // x = sign(cos)*|cos|^(2/p), y = sign(sin)*|sin|^(2/p)
                                const x = Math.sign(cosT) * Math.pow(Math.abs(cosT), 2 / p);
                                const y = Math.sign(sinT) * Math.pow(Math.abs(sinT), 2 / p);
                                const [sx, sy] = viz.toScreen(x, y);
                                if (i === 0) ctx.moveTo(sx, sy);
                                else ctx.lineTo(sx, sy);
                            }
                            ctx.closePath();
                            ctx.fillStyle = viz.colors.teal + '18';
                            ctx.fill();
                            ctx.strokeStyle = viz.colors.teal;
                            ctx.lineWidth = 2.5;
                            ctx.stroke();

                            // Mark the points on axes
                            viz.drawPoint(1, 0, viz.colors.orange, '', 4);
                            viz.drawPoint(-1, 0, viz.colors.orange, '', 4);
                            viz.drawPoint(0, 1, viz.colors.orange, '', 4);
                            viz.drawPoint(0, -1, viz.colors.orange, '', 4);

                            // Label
                            const pLabel = p >= 10 ? '\u221e' : p.toFixed(1);
                            viz.screenText('||x||' + pLabel + ' = 1', viz.width / 2, 24, viz.colors.white, 14);

                            // Show norm name
                            let name = '';
                            if (Math.abs(p - 1) < 0.05) name = 'Manhattan / L\u00b9 (promotes sparsity)';
                            else if (Math.abs(p - 2) < 0.05) name = 'Euclidean / L\u00b2 (rotation invariant)';
                            else if (p >= 9.9) name = 'Max / L\u221e (hypercube)';
                            else if (p < 1) name = 'p < 1: non-convex (not a true norm)';
                            if (name) {
                                viz.screenText(name, viz.width / 2, viz.height - 18, viz.colors.yellow, 12);
                            }
                        }

                        viz.animate(draw);
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Compute \\(\\|\\mathbf{x}\\|_1\\), \\(\\|\\mathbf{x}\\|_2\\), and \\(\\|\\mathbf{x}\\|_\\infty\\) for \\(\\mathbf{x} = (3, -4, 0, 1)^\\top\\). Which norm is always largest? Which is always smallest?',
                    hint: 'For the ordering, recall the general inequality \\(\\|\\mathbf{x}\\|_\\infty \\leq \\|\\mathbf{x}\\|_2 \\leq \\|\\mathbf{x}\\|_1\\).',
                    solution: '\\(\\|\\mathbf{x}\\|_1 = |3| + |-4| + |0| + |1| = 8\\). \\(\\|\\mathbf{x}\\|_2 = \\sqrt{9 + 16 + 0 + 1} = \\sqrt{26} \\approx 5.10\\). \\(\\|\\mathbf{x}\\|_\\infty = \\max(3, 4, 0, 1) = 4\\). In general, \\(\\|\\mathbf{x}\\|_\\infty \\leq \\|\\mathbf{x}\\|_2 \\leq \\|\\mathbf{x}\\|_1\\), so the \\(L^1\\) norm is always largest and the \\(L^\\infty\\) norm is always smallest.'
                },
                {
                    question: 'Explain geometrically why L1 regularization (adding \\(\\lambda \\|\\mathbf{w}\\|_1\\) to the loss) tends to produce sparse weight vectors, while L2 regularization (adding \\(\\lambda \\|\\mathbf{w}\\|_2^2\\)) does not.',
                    hint: 'Think about the shape of the constraint region and where the level curves of the loss function are likely to touch it.',
                    solution: 'The L1 constraint set \\(\\{\\mathbf{w} : \\|\\mathbf{w}\\|_1 \\leq t\\}\\) is a diamond with sharp corners on the coordinate axes. The level curves of a typical smooth loss function are ellipsoidal. The tangent point between an ellipse and a diamond is most likely to occur at a corner, which corresponds to one or more components being exactly zero. In contrast, the L2 ball is a smooth sphere with no corners, so tangent points generically occur at non-axis locations where no component is exactly zero.'
                },
                {
                    question: 'Show that the Frobenius norm satisfies \\(\\|\\mathbf{A}\\|_F^2 = \\operatorname{tr}(\\mathbf{A}^\\top \\mathbf{A}) = \\sum_{i} \\sigma_i^2\\), where \\(\\sigma_i\\) are the singular values of \\(\\mathbf{A}\\).',
                    hint: 'Use the SVD \\(\\mathbf{A} = \\mathbf{U}\\mathbf{\\Sigma}\\mathbf{V}^\\top\\) and the cyclic property of the trace: \\(\\operatorname{tr}(\\mathbf{X}\\mathbf{Y}\\mathbf{Z}) = \\operatorname{tr}(\\mathbf{Z}\\mathbf{X}\\mathbf{Y})\\).',
                    solution: 'By definition, \\(\\|\\mathbf{A}\\|_F^2 = \\sum_{ij} a_{ij}^2 = \\operatorname{tr}(\\mathbf{A}^\\top \\mathbf{A})\\). Substituting the SVD: \\(\\mathbf{A}^\\top \\mathbf{A} = \\mathbf{V}\\mathbf{\\Sigma}^\\top \\mathbf{U}^\\top \\mathbf{U}\\mathbf{\\Sigma}\\mathbf{V}^\\top = \\mathbf{V}\\mathbf{\\Sigma}^2 \\mathbf{V}^\\top\\). By the cyclic property, \\(\\operatorname{tr}(\\mathbf{V}\\mathbf{\\Sigma}^2 \\mathbf{V}^\\top) = \\operatorname{tr}(\\mathbf{\\Sigma}^2 \\mathbf{V}^\\top \\mathbf{V}) = \\operatorname{tr}(\\mathbf{\\Sigma}^2) = \\sum_i \\sigma_i^2\\).'
                }
            ]
        },

        // ========== SECTION 3: Eigendecomposition ==========
        {
            id: 'sec03-eigendecomposition',
            title: 'Eigendecomposition',
            content: `
<h2>0.3 Eigendecomposition</h2>

<p>Eigendecomposition reveals the intrinsic axes along which a matrix acts by simple scaling. Understanding eigenvalues and eigenvectors is essential for analyzing optimization landscapes (Hessian eigenvalues determine curvature), understanding PCA (which is eigendecomposition of the covariance matrix), and studying the dynamics of recurrent neural networks.</p>

<h3>Eigenvalues and Eigenvectors</h3>

<div class="env-block definition">
<strong>Definition 0.3.1 (Eigenvector and Eigenvalue).</strong> A nonzero vector \\(\\mathbf{v} \\in \\mathbb{R}^n\\) is an <em>eigenvector</em> of a square matrix \\(\\mathbf{A} \\in \\mathbb{R}^{n \\times n}\\) with corresponding <em>eigenvalue</em> \\(\\lambda \\in \\mathbb{R}\\) (or \\(\\mathbb{C}\\)) if
\\[
\\mathbf{A}\\mathbf{v} = \\lambda \\mathbf{v}.
\\]
The matrix stretches (or compresses, or flips) the eigenvector \\(\\mathbf{v}\\) by the factor \\(\\lambda\\), without changing its direction.
</div>

<div class="env-block intuition">
<strong>Geometric Meaning.</strong> Most vectors change direction when multiplied by a matrix. Eigenvectors are special: they only get scaled. In the visualization below, drag a vector and watch how the matrix transforms it. The eigenvectors (shown as dashed lines) are the directions that remain on their own line after transformation. The eigenvalue tells you the scaling factor along that direction.
</div>

<div class="viz-placeholder" data-viz="viz-eigen"></div>

<h3>The Characteristic Polynomial</h3>

<p>Eigenvalues are the roots of the <em>characteristic polynomial</em>:</p>

\\[
\\det(\\mathbf{A} - \\lambda \\mathbf{I}) = 0.
\\]

<p>For a \\(2 \\times 2\\) matrix \\(\\mathbf{A} = \\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}\\), this gives the quadratic</p>

\\[
\\lambda^2 - (a + d)\\lambda + (ad - bc) = 0,
\\]

<p>so \\(\\lambda = \\frac{(a+d) \\pm \\sqrt{(a+d)^2 - 4(ad-bc)}}{2}\\). The eigenvalues can be real or complex.</p>

<h3>Diagonalization</h3>

<div class="env-block theorem">
<strong>Theorem 0.3.2 (Eigendecomposition).</strong> If \\(\\mathbf{A} \\in \\mathbb{R}^{n \\times n}\\) has \\(n\\) linearly independent eigenvectors \\(\\mathbf{v}_1, \\ldots, \\mathbf{v}_n\\) with eigenvalues \\(\\lambda_1, \\ldots, \\lambda_n\\), then
\\[
\\mathbf{A} = \\mathbf{V} \\boldsymbol{\\Lambda} \\mathbf{V}^{-1},
\\]
where \\(\\mathbf{V} = [\\mathbf{v}_1 | \\cdots | \\mathbf{v}_n]\\) and \\(\\boldsymbol{\\Lambda} = \\operatorname{diag}(\\lambda_1, \\ldots, \\lambda_n)\\). In this form, \\(\\mathbf{A}\\) acts by: (1) change to the eigenvector basis (\\(\\mathbf{V}^{-1}\\)), (2) scale each axis by its eigenvalue (\\(\\boldsymbol{\\Lambda}\\)), and (3) change back (\\(\\mathbf{V}\\)).
</div>

<div class="env-block theorem">
<strong>Theorem 0.3.3 (Spectral Theorem).</strong> If \\(\\mathbf{A}\\) is <em>real symmetric</em> (\\(\\mathbf{A} = \\mathbf{A}^\\top\\)), then:
<ol>
<li>All eigenvalues are real.</li>
<li>Eigenvectors corresponding to distinct eigenvalues are orthogonal.</li>
<li>\\(\\mathbf{A}\\) can be diagonalized as \\(\\mathbf{A} = \\mathbf{Q} \\boldsymbol{\\Lambda} \\mathbf{Q}^\\top\\), where \\(\\mathbf{Q}\\) is an orthogonal matrix (\\(\\mathbf{Q}^{-1} = \\mathbf{Q}^\\top\\)).</li>
</ol>
This is enormously important: the Hessian of a loss function is symmetric, so its eigendecomposition always exists and has real eigenvalues. The eigenvalues of the Hessian are the curvatures along the principal directions of the loss surface.
</div>

<div class="env-block example">
<strong>Example 0.3.4 (PCA as Eigendecomposition).</strong> Principal Component Analysis (PCA) computes the eigendecomposition of the covariance matrix \\(\\mathbf{\\Sigma} = \\frac{1}{n}\\mathbf{X}^\\top \\mathbf{X}\\). The eigenvectors are the principal directions (axes of maximum variance), and the eigenvalues are the variances along those axes. Keeping only the top \\(k\\) eigenvectors gives the best rank-\\(k\\) approximation of the data.
</div>
`,
            visualizations: [
                {
                    id: 'viz-eigen',
                    title: 'Eigenvectors Stay on Their Line',
                    description: 'Drag the blue vector. The orange vector shows \\(\\mathbf{A}\\mathbf{v}\\). Eigenvector directions (dashed lines) are the special directions where \\(\\mathbf{A}\\mathbf{v}\\) stays collinear with \\(\\mathbf{v}\\). Adjust the matrix to see how eigenvectors change.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 55 });
                        let ma = 2, mb = 1, mc = 1, md = 3;

                        VizEngine.createSlider(controls, 'a', -3, 4, 2, 0.1, v => { ma = v; });
                        VizEngine.createSlider(controls, 'b', -3, 3, 1, 0.1, v => { mb = v; });
                        VizEngine.createSlider(controls, 'c', -3, 3, 1, 0.1, v => { mc = v; });
                        VizEngine.createSlider(controls, 'd', -3, 4, 3, 0.1, v => { md = v; });

                        const dragV = viz.addDraggable('v', 1.5, 1.0, viz.colors.blue, 8, (wx, wy) => {
                            const len = Math.sqrt(wx * wx + wy * wy);
                            if (len < 0.1) { dragV.x = 0.1; dragV.y = 0; return; }
                            // Clamp magnitude
                            if (len > 3.5) { dragV.x = wx / len * 3.5; dragV.y = wy / len * 3.5; }
                        });

                        function draw() {
                            viz.clear();
                            viz.drawGrid();
                            viz.drawAxes();

                            const M = [[ma, mb], [mc, md]];
                            const vx = dragV.x, vy = dragV.y;

                            // Compute eigenvectors
                            const evals = VizEngine.eigenvalues2(M);
                            if (evals) {
                                for (let idx = 0; idx < 2; idx++) {
                                    const lam = evals[idx];
                                    const ev = VizEngine.eigenvector2(M, lam);
                                    const color = idx === 0 ? viz.colors.green : viz.colors.purple;
                                    // Draw eigenvector line (extends to infinity)
                                    viz.drawLine(0, 0, ev[0], ev[1], color + '55', 1.5, true);
                                    // Draw eigenvector arrow
                                    viz.drawVector(0, 0, ev[0] * 1.5, ev[1] * 1.5, color, '\u03BB=' + lam.toFixed(2), 1.5);
                                }
                            } else {
                                viz.screenText('Complex eigenvalues (rotation component)', viz.width / 2, viz.height - 18, viz.colors.yellow, 12);
                            }

                            // Original vector
                            viz.drawVector(0, 0, vx, vy, viz.colors.blue, 'v', 2.5);

                            // Transformed vector
                            const tv = VizEngine.matVec(M, [vx, vy]);
                            // Clamp display if too large
                            const tvLen = VizEngine.vecLen(tv);
                            let tvx = tv[0], tvy = tv[1];
                            if (tvLen > 5) {
                                tvx = tv[0] / tvLen * 5;
                                tvy = tv[1] / tvLen * 5;
                            }
                            viz.drawVector(0, 0, tvx, tvy, viz.colors.orange, 'Av', 2.5);

                            // Show angle between v and Av
                            const origAngle = Math.atan2(vy, vx);
                            const transAngle = Math.atan2(tvy, tvx);
                            let angleDiff = (transAngle - origAngle) * 180 / Math.PI;
                            if (angleDiff > 180) angleDiff -= 360;
                            if (angleDiff < -180) angleDiff += 360;

                            viz.screenText(
                                'A = [' + ma.toFixed(1) + ', ' + mb.toFixed(1) + '; ' + mc.toFixed(1) + ', ' + md.toFixed(1) + ']',
                                viz.width / 2, 20, viz.colors.white, 13
                            );
                            viz.screenText(
                                'Direction change: ' + Math.abs(angleDiff).toFixed(1) + '\u00b0' + (Math.abs(angleDiff) < 2 ? '  \u2190 on eigenvector!' : ''),
                                viz.width / 2, 40, Math.abs(angleDiff) < 2 ? viz.colors.green : viz.colors.text, 12
                            );

                            viz.drawDraggables();
                        }

                        viz.animate(draw);
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Find the eigenvalues and eigenvectors of \\(\\mathbf{A} = \\begin{bmatrix} 4 & 2 \\\\ 1 & 3 \\end{bmatrix}\\).',
                    hint: 'The characteristic polynomial is \\(\\lambda^2 - 7\\lambda + 10 = 0\\). Factor it.',
                    solution: 'The characteristic polynomial is \\(\\lambda^2 - (4+3)\\lambda + (12-2) = \\lambda^2 - 7\\lambda + 10 = (\\lambda - 5)(\\lambda - 2) = 0\\). So \\(\\lambda_1 = 5\\) and \\(\\lambda_2 = 2\\). For \\(\\lambda_1 = 5\\): \\((\\mathbf{A} - 5\\mathbf{I})\\mathbf{v} = \\mathbf{0}\\) gives \\(-v_1 + 2v_2 = 0\\), so \\(\\mathbf{v}_1 = (2, 1)^\\top\\). For \\(\\lambda_2 = 2\\): \\((\\mathbf{A} - 2\\mathbf{I})\\mathbf{v} = \\mathbf{0}\\) gives \\(2v_1 + 2v_2 = 0\\), so \\(\\mathbf{v}_2 = (1, -1)^\\top\\).'
                },
                {
                    question: 'The Hessian of a loss function at a critical point has eigenvalues \\(\\{0.01, 0.5, 100\\}\\). What does this tell you about the optimization landscape? Why might gradient descent struggle here?',
                    hint: 'Think about the condition number \\(\\kappa = \\lambda_{\\max} / \\lambda_{\\min}\\) and what it means for the shape of level curves.',
                    solution: 'The eigenvalues represent curvatures along the principal axes. The condition number is \\(\\kappa = 100 / 0.01 = 10{,}000\\), indicating the loss surface is extremely elongated (like a narrow valley). The gradient points "across" the valley rather than "along" it, causing oscillations with large learning rates and extremely slow progress with small ones. This is the classic ill-conditioning problem. Adaptive optimizers (Adam, RMSProp) help by effectively preconditioning with a diagonal approximation of the Hessian.'
                },
                {
                    question: 'Prove that a real symmetric matrix \\(\\mathbf{A}\\) has only real eigenvalues.',
                    hint: 'Suppose \\(\\mathbf{A}\\mathbf{v} = \\lambda \\mathbf{v}\\) where \\(\\lambda\\) and \\(\\mathbf{v}\\) may be complex. Consider \\(\\overline{\\mathbf{v}}^\\top \\mathbf{A} \\mathbf{v}\\) and use \\(\\mathbf{A} = \\mathbf{A}^\\top\\) and \\(\\mathbf{A}\\) is real.',
                    solution: 'Let \\(\\mathbf{A}\\mathbf{v} = \\lambda \\mathbf{v}\\) with \\(\\mathbf{v} \\neq \\mathbf{0}\\). Then \\(\\bar{\\mathbf{v}}^\\top \\mathbf{A} \\mathbf{v} = \\lambda \\bar{\\mathbf{v}}^\\top \\mathbf{v} = \\lambda \\|\\mathbf{v}\\|^2\\). Taking the conjugate transpose: \\(\\overline{\\bar{\\mathbf{v}}^\\top \\mathbf{A} \\mathbf{v}} = (\\mathbf{A}\\mathbf{v})^* \\mathbf{v} = \\bar{\\lambda} \\bar{\\mathbf{v}}^\\top \\mathbf{v} = \\bar{\\lambda} \\|\\mathbf{v}\\|^2\\). But \\(\\mathbf{A}\\) is real and symmetric, so \\(\\bar{\\mathbf{v}}^\\top \\mathbf{A} \\mathbf{v}\\) is a real number (since \\(\\bar{\\mathbf{v}}^\\top \\mathbf{A} \\mathbf{v} = \\overline{\\bar{\\mathbf{v}}^\\top \\mathbf{A} \\mathbf{v}}\\)), which forces \\(\\lambda = \\bar{\\lambda}\\), i.e., \\(\\lambda\\) is real.'
                }
            ]
        },

        // ========== SECTION 4: Singular Value Decomposition ==========
        {
            id: 'sec04-svd',
            title: 'Singular Value Decomposition',
            content: `
<h2>0.4 Singular Value Decomposition</h2>

<p>The Singular Value Decomposition (SVD) is arguably the most important matrix factorization in applied mathematics. Unlike eigendecomposition, which requires a square matrix and may not exist, the SVD exists for <em>every</em> matrix (rectangular or square, any rank). It reveals the fundamental geometry of a linear transformation: every linear map is a rotation, followed by axis-aligned scaling, followed by another rotation.</p>

<h3>The SVD Factorization</h3>

<div class="env-block theorem">
<strong>Theorem 0.4.1 (Singular Value Decomposition).</strong> Any matrix \\(\\mathbf{A} \\in \\mathbb{R}^{m \\times n}\\) can be factored as
\\[
\\mathbf{A} = \\mathbf{U} \\boldsymbol{\\Sigma} \\mathbf{V}^\\top,
\\]
where:
<ul>
<li>\\(\\mathbf{U} \\in \\mathbb{R}^{m \\times m}\\) is orthogonal: its columns \\(\\mathbf{u}_i\\) are the <em>left singular vectors</em>.</li>
<li>\\(\\boldsymbol{\\Sigma} \\in \\mathbb{R}^{m \\times n}\\) is diagonal with non-negative entries \\(\\sigma_1 \\geq \\sigma_2 \\geq \\cdots \\geq 0\\), the <em>singular values</em>.</li>
<li>\\(\\mathbf{V} \\in \\mathbb{R}^{n \\times n}\\) is orthogonal: its columns \\(\\mathbf{v}_i\\) are the <em>right singular vectors</em>.</li>
</ul>
</div>

<div class="env-block intuition">
<strong>Three-Step Geometry.</strong> The SVD decomposes any linear transformation into three stages:
<ol>
<li><strong>\\(\\mathbf{V}^\\top\\):</strong> Rotate (or reflect) in the input space to align with the "natural" axes of the transformation.</li>
<li><strong>\\(\\boldsymbol{\\Sigma}\\):</strong> Scale each axis independently by the corresponding singular value. This is where the transformation's "essence" lives.</li>
<li><strong>\\(\\mathbf{U}\\):</strong> Rotate (or reflect) in the output space to the final orientation.</li>
</ol>
The unit circle (or sphere) in the input space maps to an ellipse (or ellipsoid) in the output space. The semi-axes of this ellipse have lengths equal to the singular values, and they point in the directions of the left singular vectors.
</div>

<div class="viz-placeholder" data-viz="viz-svd"></div>

<h3>Low-Rank Approximation</h3>

<div class="env-block theorem">
<strong>Theorem 0.4.2 (Eckart-Young-Mirsky).</strong> The best rank-\\(k\\) approximation of \\(\\mathbf{A}\\) (in both Frobenius and spectral norms) is obtained by keeping only the top \\(k\\) singular values:
\\[
\\mathbf{A}_k = \\sum_{i=1}^k \\sigma_i \\mathbf{u}_i \\mathbf{v}_i^\\top.
\\]
The approximation error is \\(\\|\\mathbf{A} - \\mathbf{A}_k\\|_F^2 = \\sum_{i=k+1}^r \\sigma_i^2\\).
</div>

<div class="env-block example">
<strong>Example 0.4.3 (LoRA: Low-Rank Adaptation).</strong> Fine-tuning large language models is expensive because the weight matrices have millions of parameters. LoRA (Low-Rank Adaptation) freezes the pretrained weights \\(\\mathbf{W}_0\\) and learns a low-rank update:
\\[
\\mathbf{W} = \\mathbf{W}_0 + \\mathbf{B}\\mathbf{A},
\\]
where \\(\\mathbf{B} \\in \\mathbb{R}^{d \\times r}\\) and \\(\\mathbf{A} \\in \\mathbb{R}^{r \\times d}\\) with \\(r \\ll d\\). The intuition from the Eckart-Young theorem is that the weight update \\(\\Delta \\mathbf{W}\\) for a specific task likely has low intrinsic rank, so a rank-\\(r\\) factorization can capture most of the adaptation with far fewer parameters.
</div>

<div class="env-block remark">
<strong>Connection to Eigendecomposition.</strong> The singular values of \\(\\mathbf{A}\\) are the square roots of the eigenvalues of \\(\\mathbf{A}^\\top \\mathbf{A}\\) (or \\(\\mathbf{A}\\mathbf{A}^\\top\\)). The right singular vectors \\(\\mathbf{v}_i\\) are eigenvectors of \\(\\mathbf{A}^\\top \\mathbf{A}\\), and the left singular vectors \\(\\mathbf{u}_i\\) are eigenvectors of \\(\\mathbf{A}\\mathbf{A}^\\top\\). For a symmetric positive semi-definite matrix, the SVD and eigendecomposition coincide.
</div>
`,
            visualizations: [
                {
                    id: 'viz-svd',
                    title: 'SVD: Three-Step Geometric Decomposition',
                    description: 'Watch the SVD decomposition step by step. The unit circle (blue) is first rotated by \\(\\mathbf{V}^\\top\\) (green), then scaled by \\(\\boldsymbol{\\Sigma}\\) (yellow), then rotated by \\(\\mathbf{U}\\) (orange) to produce the final ellipse. Use the slider to animate through the stages or adjust the matrix.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 55 });
                        let t = 0;  // animation parameter: 0->1 VT, 1->2 Sigma, 2->3 U
                        let ma = 2, mb = 1, mc = 0.5, md = 1.5;

                        const tSlider = VizEngine.createSlider(controls, 'Stage', 0, 3, 0, 0.02, v => { t = v; });
                        VizEngine.createSlider(controls, 'a', -3, 3, 2, 0.1, v => { ma = v; });
                        VizEngine.createSlider(controls, 'b', -3, 3, 1, 0.1, v => { mb = v; });
                        VizEngine.createSlider(controls, 'c', -3, 3, 0.5, 0.1, v => { mc = v; });
                        VizEngine.createSlider(controls, 'd', -3, 3, 1.5, 0.1, v => { md = v; });

                        // Compute SVD of 2x2 matrix
                        function svd2x2(a, b, c, d) {
                            // A^T A
                            const ata00 = a*a + c*c, ata01 = a*b + c*d;
                            const ata10 = ata01, ata11 = b*b + d*d;
                            // Eigenvalues of A^T A
                            const tr = ata00 + ata11;
                            const det = ata00 * ata11 - ata01 * ata10;
                            const disc = Math.max(0, tr*tr - 4*det);
                            const s = Math.sqrt(disc);
                            const lam1 = (tr + s) / 2, lam2 = Math.max(0, (tr - s) / 2);
                            const sig1 = Math.sqrt(lam1), sig2 = Math.sqrt(lam2);

                            // Eigenvectors of A^T A -> V
                            let v1, v2;
                            if (Math.abs(ata01) > 1e-10) {
                                v1 = VizEngine.normalize([ata01, lam1 - ata00]);
                                v2 = [-v1[1], v1[0]];
                            } else {
                                v1 = ata00 >= ata11 ? [1, 0] : [0, 1];
                                v2 = ata00 >= ata11 ? [0, 1] : [1, 0];
                            }

                            // U columns: u_i = A v_i / sigma_i
                            let u1, u2;
                            if (sig1 > 1e-10) {
                                u1 = VizEngine.normalize([a*v1[0]+b*v1[1], c*v1[0]+d*v1[1]]);
                            } else {
                                u1 = [1, 0];
                            }
                            if (sig2 > 1e-10) {
                                u2 = VizEngine.normalize([a*v2[0]+b*v2[1], c*v2[0]+d*v2[1]]);
                            } else {
                                u2 = [-u1[1], u1[0]];
                            }

                            // Ensure right-handed
                            if (v1[0]*v2[1] - v1[1]*v2[0] < 0) { v2 = [-v2[0], -v2[1]]; }
                            if (u1[0]*u2[1] - u1[1]*u2[0] < 0) { u2 = [-u2[0], -u2[1]]; }

                            return {
                                U: [[u1[0], u2[0]], [u1[1], u2[1]]],
                                S: [sig1, sig2],
                                V: [[v1[0], v2[0]], [v1[1], v2[1]]]
                            };
                        }

                        function draw() {
                            viz.clear();
                            viz.drawGrid();
                            viz.drawAxes();

                            const svd = svd2x2(ma, mb, mc, md);
                            const { U, S, V } = svd;

                            // VT = V transposed
                            const VT = [[V[0][0], V[1][0]], [V[0][1], V[1][1]]];

                            const N = 72;
                            const ctx = viz.ctx;

                            // Helper: interpolate matrices
                            function interpMat(M1, M2, frac) {
                                return [
                                    [M1[0][0]*(1-frac)+M2[0][0]*frac, M1[0][1]*(1-frac)+M2[0][1]*frac],
                                    [M1[1][0]*(1-frac)+M2[1][0]*frac, M1[1][1]*(1-frac)+M2[1][1]*frac]
                                ];
                            }

                            // Identity
                            const I = [[1,0],[0,1]];
                            // Sigma as matrix
                            const SM = [[S[0], 0], [0, S[1]]];
                            // Sigma * VT
                            const SVT = VizEngine.matMul(SM, VT);
                            // Full: U * Sigma * VT
                            const USVT = VizEngine.matMul(U, SVT);

                            // Current transformation matrix based on t
                            let M;
                            let stageName;
                            if (t <= 1) {
                                // Stage 1: Identity -> VT
                                M = interpMat(I, VT, t);
                                stageName = 'Stage 1: Applying V\u1d40 (rotate input)';
                            } else if (t <= 2) {
                                // Stage 2: VT -> Sigma * VT
                                M = interpMat(VT, SVT, t - 1);
                                stageName = 'Stage 2: Applying \u03a3 (scale axes)';
                            } else {
                                // Stage 3: Sigma * VT -> U * Sigma * VT
                                M = interpMat(SVT, USVT, t - 2);
                                stageName = 'Stage 3: Applying U (rotate output)';
                            }

                            // Draw original unit circle (faint)
                            ctx.beginPath();
                            for (let i = 0; i <= N; i++) {
                                const ang = (2 * Math.PI * i) / N;
                                const [sx, sy] = viz.toScreen(Math.cos(ang), Math.sin(ang));
                                i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                            }
                            ctx.closePath();
                            ctx.strokeStyle = viz.colors.blue + '44';
                            ctx.lineWidth = 1;
                            ctx.stroke();

                            // Draw transformed circle
                            ctx.beginPath();
                            for (let i = 0; i <= N; i++) {
                                const ang = (2 * Math.PI * i) / N;
                                const px = Math.cos(ang), py = Math.sin(ang);
                                const tx = M[0][0]*px + M[0][1]*py;
                                const ty = M[1][0]*px + M[1][1]*py;
                                const [sx, sy] = viz.toScreen(tx, ty);
                                i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                            }
                            ctx.closePath();
                            ctx.fillStyle = viz.colors.orange + '15';
                            ctx.fill();
                            ctx.strokeStyle = viz.colors.orange;
                            ctx.lineWidth = 2;
                            ctx.stroke();

                            // Draw transformed basis vectors
                            const te1 = VizEngine.matVec(M, [1, 0]);
                            const te2 = VizEngine.matVec(M, [0, 1]);
                            viz.drawVector(0, 0, te1[0], te1[1], viz.colors.teal, '', 2);
                            viz.drawVector(0, 0, te2[0], te2[1], viz.colors.green, '', 2);

                            // Labels
                            viz.screenText(stageName, viz.width / 2, 22, viz.colors.white, 13);
                            viz.screenText(
                                '\u03c3\u2081 = ' + S[0].toFixed(2) + ',  \u03c3\u2082 = ' + S[1].toFixed(2),
                                viz.width / 2, viz.height - 18, viz.colors.yellow, 12
                            );
                        }

                        viz.animate(draw);
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Compute the SVD of \\(\\mathbf{A} = \\begin{bmatrix} 3 & 0 \\\\ 0 & -2 \\end{bmatrix}\\). What are the singular values, and why are they both positive even though \\(\\mathbf{A}\\) has a negative entry?',
                    hint: 'For a diagonal matrix, the singular values are the absolute values of the diagonal entries. Think about what \\(\\mathbf{U}\\) must do to "absorb" the sign.',
                    solution: 'Since \\(\\mathbf{A}\\) is diagonal, \\(\\mathbf{A}^\\top \\mathbf{A} = \\begin{bmatrix} 9 & 0 \\\\ 0 & 4 \\end{bmatrix}\\), so \\(\\sigma_1 = 3\\) and \\(\\sigma_2 = 2\\). We have \\(\\mathbf{V} = \\mathbf{I}\\) and \\(\\boldsymbol{\\Sigma} = \\begin{bmatrix} 3 & 0 \\\\ 0 & 2 \\end{bmatrix}\\). Since \\(\\mathbf{u}_2 = \\mathbf{A}\\mathbf{v}_2 / \\sigma_2 = (0, -2)^\\top / 2 = (0, -1)^\\top\\), we get \\(\\mathbf{U} = \\begin{bmatrix} 1 & 0 \\\\ 0 & -1 \\end{bmatrix}\\). The negative sign in \\(\\mathbf{A}\\) is absorbed by \\(\\mathbf{U}\\), keeping all singular values non-negative. Singular values measure "stretching magnitude" and cannot be negative.'
                },
                {
                    question: 'A weight matrix \\(\\mathbf{W} \\in \\mathbb{R}^{1000 \\times 1000}\\) has singular values that decay rapidly: \\(\\sigma_1 = 10\\), \\(\\sigma_2 = 5\\), \\(\\sigma_{10} = 0.1\\), and \\(\\sigma_k < 0.01\\) for \\(k > 20\\). How many parameters does a rank-20 approximation require compared to the full matrix? What fraction of \\(\\|\\mathbf{W}\\|_F^2\\) is captured?',
                    hint: 'A rank-\\(r\\) factorization \\(\\mathbf{W} \\approx \\mathbf{B}\\mathbf{C}\\) with \\(\\mathbf{B} \\in \\mathbb{R}^{m \\times r}\\) and \\(\\mathbf{C} \\in \\mathbb{R}^{r \\times n}\\) has \\(r(m+n)\\) parameters.',
                    solution: 'The full matrix has \\(1000 \\times 1000 = 10^6\\) parameters. A rank-20 factorization has \\(20 \\times (1000 + 1000) = 40{,}000\\) parameters, which is 4% of the original. The energy captured is \\(\\sum_{i=1}^{20} \\sigma_i^2 / \\|\\mathbf{W}\\|_F^2\\). Since \\(\\sigma_k < 0.01\\) for \\(k > 20\\), the tail contributes at most \\(980 \\times 0.01^2 = 0.098\\), while the top 20 contribute the vast majority. This is the principle behind LoRA and other low-rank compression methods.'
                },
                {
                    question: 'Prove that the rank of \\(\\mathbf{A}\\) equals the number of nonzero singular values.',
                    hint: 'The rank is the dimension of the column space. Use the SVD to express the column space in terms of the left singular vectors.',
                    solution: 'Let \\(\\mathbf{A} = \\mathbf{U}\\boldsymbol{\\Sigma}\\mathbf{V}^\\top\\) with \\(r\\) nonzero singular values. For any \\(\\mathbf{x} \\in \\mathbb{R}^n\\), \\(\\mathbf{A}\\mathbf{x} = \\mathbf{U}\\boldsymbol{\\Sigma}(\\mathbf{V}^\\top \\mathbf{x}) = \\sum_{i=1}^r \\sigma_i (\\mathbf{v}_i^\\top \\mathbf{x}) \\mathbf{u}_i\\). Thus the column space of \\(\\mathbf{A}\\) is \\(\\operatorname{span}\\{\\mathbf{u}_1, \\ldots, \\mathbf{u}_r\\}\\), which has dimension \\(r\\) since the \\(\\mathbf{u}_i\\) are orthonormal. Therefore \\(\\operatorname{rank}(\\mathbf{A}) = r\\), the number of nonzero singular values.'
                }
            ]
        },

        // ========== SECTION 5: Special Matrices in Deep Learning ==========
        {
            id: 'sec05-special-matrices',
            title: 'Special Matrices in DL',
            content: `
<h2>0.5 Special Matrices in Deep Learning</h2>

<p>Certain classes of matrices appear repeatedly in deep learning theory and practice. Understanding their properties helps you reason about optimization (positive definite Hessians), stability (orthogonal weight matrices), and statistical structure (symmetric covariance matrices).</p>

<h3>Symmetric Matrices</h3>

<div class="env-block definition">
<strong>Definition 0.5.1 (Symmetric Matrix).</strong> A matrix \\(\\mathbf{A} \\in \\mathbb{R}^{n \\times n}\\) is <em>symmetric</em> if \\(\\mathbf{A} = \\mathbf{A}^\\top\\). Equivalently, \\(a_{ij} = a_{ji}\\) for all \\(i, j\\). The key consequence (Spectral Theorem, Theorem 0.3.3) is that symmetric matrices always have real eigenvalues and orthogonal eigenvectors.
</div>

<div class="env-block remark">
<strong>Where Symmetric Matrices Appear.</strong> Covariance matrices \\(\\boldsymbol{\\Sigma} = \\mathbb{E}[(\\mathbf{x} - \\boldsymbol{\\mu})(\\mathbf{x} - \\boldsymbol{\\mu})^\\top]\\) are always symmetric. The Hessian \\(\\mathbf{H} = \\nabla^2 f\\) of any twice-differentiable function is symmetric (by equality of mixed partials). The Gram matrix \\(\\mathbf{X}^\\top \\mathbf{X}\\) is always symmetric. The Fisher information matrix is symmetric.
</div>

<h3>Positive Definite and Positive Semi-Definite Matrices</h3>

<div class="env-block definition">
<strong>Definition 0.5.2 (Positive Definiteness).</strong> A symmetric matrix \\(\\mathbf{A} \\in \\mathbb{R}^{n \\times n}\\) is:
<ul>
<li><em>Positive definite</em> (\\(\\mathbf{A} \\succ 0\\)) if \\(\\mathbf{x}^\\top \\mathbf{A} \\mathbf{x} &gt; 0\\) for all \\(\\mathbf{x} \\neq \\mathbf{0}\\).</li>
<li><em>Positive semi-definite</em> (\\(\\mathbf{A} \\succeq 0\\)) if \\(\\mathbf{x}^\\top \\mathbf{A} \\mathbf{x} \\geq 0\\) for all \\(\\mathbf{x}\\).</li>
</ul>
Equivalently, \\(\\mathbf{A} \\succ 0\\) if and only if all eigenvalues are strictly positive.
</div>

<div class="env-block intuition">
<strong>Positive Definiteness and Optimization.</strong> The quadratic form \\(f(\\mathbf{x}) = \\frac{1}{2}\\mathbf{x}^\\top \\mathbf{A} \\mathbf{x} - \\mathbf{b}^\\top \\mathbf{x}\\) is a "bowl" when \\(\\mathbf{A} \\succ 0\\), with a unique global minimum at \\(\\mathbf{x}^* = \\mathbf{A}^{-1}\\mathbf{b}\\). The contours of this quadratic are ellipses whose axes are aligned with the eigenvectors of \\(\\mathbf{A}\\) and whose radii are inversely proportional to the square roots of the eigenvalues. A well-conditioned matrix (eigenvalues of similar magnitude) gives nearly circular contours and fast convergence; an ill-conditioned matrix gives highly elongated ellipses and slow, zig-zagging gradient descent.
</div>

<div class="viz-placeholder" data-viz="viz-pd-contours"></div>

<h3>Orthogonal Matrices</h3>

<div class="env-block definition">
<strong>Definition 0.5.3 (Orthogonal Matrix).</strong> A square matrix \\(\\mathbf{Q} \\in \\mathbb{R}^{n \\times n}\\) is <em>orthogonal</em> if \\(\\mathbf{Q}^\\top \\mathbf{Q} = \\mathbf{Q}\\mathbf{Q}^\\top = \\mathbf{I}\\), i.e., \\(\\mathbf{Q}^{-1} = \\mathbf{Q}^\\top\\). Orthogonal matrices preserve norms: \\(\\|\\mathbf{Q}\\mathbf{x}\\|_2 = \\|\\mathbf{x}\\|_2\\) for all \\(\\mathbf{x}\\).
</div>

<div class="env-block remark">
<strong>Orthogonal Matrices in Deep Learning.</strong>
<ul>
<li><strong>Initialization:</strong> Orthogonal initialization sets weight matrices to random orthogonal matrices, preventing gradient vanishing/exploding in the first forward pass. Since \\(\\|\\mathbf{W}\\mathbf{x}\\| = \\|\\mathbf{x}\\|\\), signals neither grow nor shrink.</li>
<li><strong>Stability in RNNs:</strong> If the recurrent weight matrix is orthogonal, the hidden state norm is preserved across time steps, mitigating the vanishing/exploding gradient problem.</li>
<li><strong>SVD components:</strong> The \\(\\mathbf{U}\\) and \\(\\mathbf{V}\\) matrices in SVD are orthogonal, representing pure rotations (or reflections).</li>
</ul>
</div>

<div class="env-block theorem">
<strong>Theorem 0.5.4 (Properties of Orthogonal Matrices).</strong> If \\(\\mathbf{Q}\\) is orthogonal, then:
<ol>
<li>\\(|\\det(\\mathbf{Q})| = 1\\) (no volume change; \\(\\det = +1\\) for rotations, \\(-1\\) for reflections).</li>
<li>All eigenvalues have absolute value 1: \\(|\\lambda_i| = 1\\).</li>
<li>\\(\\|\\mathbf{Q}\\|_2 = 1\\) (spectral norm equals 1).</li>
<li>The set of orthogonal matrices forms a group under multiplication.</li>
</ol>
</div>

<h3>Summary: Matrix Zoo for Deep Learning</h3>

<table class="data-table" style="width:100%;margin:16px 0;border-collapse:collapse;text-align:left;">
<tr style="border-bottom:2px solid #30363d;">
<th style="padding:8px 12px;">Matrix Type</th>
<th style="padding:8px 12px;">Key Property</th>
<th style="padding:8px 12px;">Where It Appears</th>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px 12px;">Symmetric</td>
<td style="padding:8px 12px;">\\(\\mathbf{A} = \\mathbf{A}^\\top\\)</td>
<td style="padding:8px 12px;">Hessian, covariance, Gram matrix</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px 12px;">Positive definite</td>
<td style="padding:8px 12px;">All eigenvalues \\(&gt; 0\\)</td>
<td style="padding:8px 12px;">Convex loss curvature, covariance</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px 12px;">Orthogonal</td>
<td style="padding:8px 12px;">\\(\\mathbf{Q}^\\top\\mathbf{Q} = \\mathbf{I}\\)</td>
<td style="padding:8px 12px;">SVD components, initialization</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px 12px;">Diagonal</td>
<td style="padding:8px 12px;">\\(a_{ij} = 0\\) for \\(i \\neq j\\)</td>
<td style="padding:8px 12px;">Singular values, batch norm scaling</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:8px 12px;">Low-rank</td>
<td style="padding:8px 12px;">\\(\\operatorname{rank} \\ll \\min(m,n)\\)</td>
<td style="padding:8px 12px;">LoRA, attention matrices, embeddings</td>
</tr>
</table>

<div class="env-block example">
<strong>Example 0.5.5 (Condition Number and Learning Rate).</strong> For the quadratic loss \\(f(\\mathbf{x}) = \\frac{1}{2}\\mathbf{x}^\\top \\mathbf{H}\\mathbf{x}\\), gradient descent with learning rate \\(\\eta\\) converges if and only if \\(\\eta &lt; 2/\\lambda_{\\max}(\\mathbf{H})\\). The convergence rate is \\(\\mathcal{O}\\left(\\left(\\frac{\\kappa - 1}{\\kappa + 1}\\right)^t\\right)\\), where \\(\\kappa = \\lambda_{\\max}/\\lambda_{\\min}\\) is the condition number. Large \\(\\kappa\\) means slow convergence, explaining why preconditioning (Adam, natural gradient) helps.
</div>
`,
            visualizations: [
                {
                    id: 'viz-pd-contours',
                    title: 'Positive Definite Matrix: Quadratic Contours',
                    description: 'Drag the control points to change the positive definite matrix \\(\\mathbf{A}\\). The elliptical contours show level sets of \\(f(\\mathbf{x}) = \\mathbf{x}^\\top\\mathbf{A}\\mathbf{x}\\). The eigenvectors (arrows) are the ellipse axes; eigenvalues determine the axis lengths. Well-conditioned matrices (\\(\\kappa \\approx 1\\)) give circular contours.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 55 });

                        // Parameterize by eigenvalues and rotation angle
                        let lam1 = 3, lam2 = 1, angle = 0.3;
                        VizEngine.createSlider(controls, '\u03bb\u2081', 0.2, 5, 3, 0.1, v => { lam1 = v; });
                        VizEngine.createSlider(controls, '\u03bb\u2082', 0.2, 5, 1, 0.1, v => { lam2 = v; });
                        VizEngine.createSlider(controls, '\u03b8', -1.57, 1.57, 0.3, 0.05, v => { angle = v; });

                        function draw() {
                            viz.clear();
                            viz.drawGrid();
                            viz.drawAxes();

                            const cos = Math.cos(angle), sin = Math.sin(angle);

                            // A = Q * diag(lam1, lam2) * Q^T where Q = [cos -sin; sin cos]
                            const a11 = lam1*cos*cos + lam2*sin*sin;
                            const a12 = (lam1-lam2)*cos*sin;
                            const a22 = lam1*sin*sin + lam2*cos*cos;

                            const ctx = viz.ctx;

                            // Draw contour ellipses
                            const levels = [0.5, 1, 2, 3, 5, 8];
                            const colors = [
                                viz.colors.teal + 'cc',
                                viz.colors.teal + 'aa',
                                viz.colors.teal + '88',
                                viz.colors.teal + '66',
                                viz.colors.teal + '44',
                                viz.colors.teal + '33'
                            ];

                            for (let li = 0; li < levels.length; li++) {
                                const level = levels[li];
                                // x^T A x = level => ellipse
                                // In eigenbasis: lam1 * u^2 + lam2 * v^2 = level
                                // u = sqrt(level/lam1)*cos(t), v = sqrt(level/lam2)*sin(t)
                                const rx = Math.sqrt(level / lam1);
                                const ry = Math.sqrt(level / lam2);

                                ctx.beginPath();
                                for (let i = 0; i <= 100; i++) {
                                    const t = (2 * Math.PI * i) / 100;
                                    const u = rx * Math.cos(t);
                                    const v = ry * Math.sin(t);
                                    // Rotate back: x = cos*u - sin*v, y = sin*u + cos*v
                                    const x = cos * u - sin * v;
                                    const y = sin * u + cos * v;
                                    const [sx, sy] = viz.toScreen(x, y);
                                    i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                                }
                                ctx.closePath();
                                ctx.strokeStyle = colors[li];
                                ctx.lineWidth = li === 0 ? 2 : 1;
                                ctx.stroke();
                            }

                            // Draw eigenvector directions
                            const ev1 = [cos, sin];
                            const ev2 = [-sin, cos];
                            viz.drawVector(0, 0, ev1[0]*2, ev1[1]*2, viz.colors.orange, '\u03bb\u2081=' + lam1.toFixed(1), 2);
                            viz.drawVector(0, 0, ev2[0]*2, ev2[1]*2, viz.colors.green, '\u03bb\u2082=' + lam2.toFixed(1), 2);

                            // Draw eigenvector lines (dashed, extended)
                            viz.drawLine(0, 0, ev1[0], ev1[1], viz.colors.orange + '33', 1, true);
                            viz.drawLine(0, 0, ev2[0], ev2[1], viz.colors.green + '33', 1, true);

                            // Mark the minimum
                            viz.drawPoint(0, 0, viz.colors.white, 'min', 4);

                            // Info
                            const kappa = Math.max(lam1, lam2) / Math.min(lam1, lam2);
                            viz.screenText(
                                'A = Q diag(' + lam1.toFixed(1) + ', ' + lam2.toFixed(1) + ') Q\u1d40     \u03ba = ' + kappa.toFixed(1),
                                viz.width / 2, 20, viz.colors.white, 13
                            );

                            let condText = '';
                            if (kappa < 1.5) condText = 'Well-conditioned: nearly circular, fast convergence';
                            else if (kappa < 5) condText = 'Moderate conditioning';
                            else condText = 'Ill-conditioned: elongated ellipses, slow convergence';
                            viz.screenText(condText, viz.width / 2, viz.height - 18, kappa < 3 ? viz.colors.green : viz.colors.yellow, 12);
                        }

                        viz.animate(draw);
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Show that \\(\\mathbf{X}^\\top \\mathbf{X}\\) is always positive semi-definite for any matrix \\(\\mathbf{X}\\). Under what conditions is it positive definite?',
                    hint: 'Compute \\(\\mathbf{z}^\\top (\\mathbf{X}^\\top \\mathbf{X}) \\mathbf{z}\\) and recognize it as a squared norm.',
                    solution: 'For any \\(\\mathbf{z}\\), \\(\\mathbf{z}^\\top \\mathbf{X}^\\top \\mathbf{X} \\mathbf{z} = (\\mathbf{X}\\mathbf{z})^\\top (\\mathbf{X}\\mathbf{z}) = \\|\\mathbf{X}\\mathbf{z}\\|_2^2 \\geq 0\\). Thus \\(\\mathbf{X}^\\top \\mathbf{X} \\succeq 0\\). It is positive definite if and only if \\(\\|\\mathbf{X}\\mathbf{z}\\|^2 > 0\\) for all \\(\\mathbf{z} \\neq \\mathbf{0}\\), which happens if and only if \\(\\mathbf{X}\\) has full column rank (i.e., the columns of \\(\\mathbf{X}\\) are linearly independent). In deep learning, this relates to whether the feature matrix has redundant features.'
                },
                {
                    question: 'A weight matrix is initialized as a random orthogonal matrix \\(\\mathbf{W}\\). Show that for any input \\(\\mathbf{x}\\), the output \\(\\mathbf{y} = \\mathbf{W}\\mathbf{x}\\) satisfies \\(\\|\\mathbf{y}\\|_2 = \\|\\mathbf{x}\\|_2\\). Why is this desirable at initialization?',
                    hint: 'Use \\(\\mathbf{W}^\\top \\mathbf{W} = \\mathbf{I}\\).',
                    solution: '\\(\\|\\mathbf{y}\\|_2^2 = \\mathbf{y}^\\top \\mathbf{y} = (\\mathbf{W}\\mathbf{x})^\\top (\\mathbf{W}\\mathbf{x}) = \\mathbf{x}^\\top \\mathbf{W}^\\top \\mathbf{W} \\mathbf{x} = \\mathbf{x}^\\top \\mathbf{I} \\mathbf{x} = \\|\\mathbf{x}\\|_2^2\\). This is desirable because at initialization, we want signals to propagate through layers without exploding (\\(\\|\\mathbf{y}\\| \\gg \\|\\mathbf{x}\\|\\)) or vanishing (\\(\\|\\mathbf{y}\\| \\ll \\|\\mathbf{x}\\|\\)). Orthogonal initialization guarantees norm preservation exactly, not just in expectation.'
                },
                {
                    question: 'The Hessian of a loss function at a critical point has eigenvalues \\(\\{2, -1, 0.5\\}\\). Is this point a local minimum, local maximum, or saddle point? What does this imply about the loss landscape of neural networks?',
                    hint: 'A local minimum requires all eigenvalues to be positive (positive definite Hessian). What happens when some are negative?',
                    solution: 'Since the Hessian has both positive (2, 0.5) and negative (-1) eigenvalues, this is a <em>saddle point</em>, not a local minimum. At a saddle point, the loss decreases in the direction of the eigenvector with eigenvalue -1 and increases in the other directions. In high-dimensional neural networks, saddle points vastly outnumber local minima. If \\(n\\) is the number of parameters, a random critical point is exponentially more likely to be a saddle point than a local minimum. This is why SGD and momentum methods, which can escape saddle points, are crucial for training.'
                }
            ]
        }
    ]
});
