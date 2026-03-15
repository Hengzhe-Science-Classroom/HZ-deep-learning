window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch09',
    number: 9,
    title: 'Convolution Operations',
    subtitle: 'Local connectivity, weight sharing, and the building blocks of convolutional neural networks',
    sections: [
        // ===================== Section 1: Convolution Intuition =====================
        {
            id: 'ch09-sec01',
            title: 'Convolution Intuition',
            content: `<h2>Convolution Intuition</h2>

                <div class="env-block intuition">
                    <div class="env-title">Why Convolution?</div>
                    <div class="env-body"><p>A fully connected layer treating a 256&times;256 RGB image as a flat vector would have \\(256 \\times 256 \\times 3 = 196{,}608\\) input units. Connecting each to even 1000 hidden units yields nearly 200 million parameters, for a single layer. This is both computationally ruinous and statistically absurd: we would need an enormous dataset just to avoid overfitting. Convolution solves this by exploiting two structural priors about images (and many other spatial or sequential signals): <strong>translation invariance</strong> and <strong>locality</strong>.</p></div>
                </div>

                <h3>Three Structural Priors</h3>

                <p>Convolutional layers encode three assumptions that dramatically reduce parameter count and improve generalization:</p>

                <ol>
                    <li><strong>Local connectivity (sparse interactions).</strong> Each output unit depends on only a small spatial neighborhood of the input, called its <em>receptive field</em>. An edge at the top-left of an image has no direct interaction with a pixel at the bottom-right.</li>
                    <li><strong>Weight sharing (parameter tying).</strong> The same set of weights (the <em>kernel</em> or <em>filter</em>) is applied at every spatial location. If a vertical-edge detector works at position \\((3,5)\\), it should work equally well at position \\((100,200)\\).</li>
                    <li><strong>Translation equivariance.</strong> If the input shifts by \\((\\Delta i, \\Delta j)\\), the output feature map shifts by the same amount. Formally, let \\(T_{\\Delta}\\) denote the translation operator. A function \\(f\\) is <em>equivariant</em> to \\(T_{\\Delta}\\) if \\(f(T_{\\Delta}[x]) = T_{\\Delta}[f(x)]\\). Convolution satisfies this property exactly.</li>
                </ol>

                <div class="env-block definition">
                    <div class="env-title">Definition (Discrete 2D Convolution)</div>
                    <div class="env-body"><p>Given an input feature map \\(X \\in \\mathbb{R}^{H \\times W}\\) and a kernel \\(K \\in \\mathbb{R}^{k_H \\times k_W}\\), the <strong>discrete 2D convolution</strong> (more precisely, cross-correlation, which is the standard convention in deep learning) produces an output</p>
                    \\[Y[i, j] = \\sum_{m=0}^{k_H - 1} \\sum_{n=0}^{k_W - 1} K[m, n] \\cdot X[i + m, j + n]\\]
                    <p>for each valid output position \\((i, j)\\). The kernel slides across the input, computing a weighted sum at each position.</p></div>
                </div>

                <div class="env-block remark">
                    <div class="env-title">Remark (Convolution vs. Cross-Correlation)</div>
                    <div class="env-body"><p>Mathematically, true convolution flips the kernel before sliding: \\((X * K)[i,j] = \\sum_m \\sum_n K[m,n] \\cdot X[i - m, j - n]\\). Deep learning frameworks implement cross-correlation instead (no flip), because the kernel weights are learned anyway, so flipping is immaterial. The literature universally calls this operation "convolution" regardless.</p></div>
                </div>

                <h3>From Fully Connected to Convolutional</h3>

                <p>Consider a 1D input of length \\(N\\) with a fully connected layer producing \\(M\\) outputs. The weight matrix \\(W \\in \\mathbb{R}^{M \\times N}\\) has \\(MN\\) free parameters. Now impose two constraints:</p>
                <ul>
                    <li><strong>Locality:</strong> Each output depends on only \\(k\\) consecutive inputs. The weight matrix becomes banded, with \\(Mk\\) nonzero entries.</li>
                    <li><strong>Weight sharing:</strong> Every row uses the same \\(k\\) weights, shifted by one position. The entire layer is parameterized by a single kernel of size \\(k\\).</li>
                </ul>

                <p>The parameter count drops from \\(MN\\) to \\(k\\), independent of the input size. For a 2D kernel of size \\(k \\times k\\) applied to an image, this is \\(k^2\\) parameters per filter, regardless of the image resolution.</p>

                <div class="env-block example">
                    <div class="env-title">Example (3&times;3 Convolution on a 5&times;5 Input)</div>
                    <div class="env-body"><p>Let the input be a \\(5 \\times 5\\) grid and the kernel be
                    \\[K = \\begin{pmatrix} 1 & 0 & -1 \\\\ 1 & 0 & -1 \\\\ 1 & 0 & -1 \\end{pmatrix}\\]
                    This is a vertical edge detector. At position \\((0,0)\\), the output is
                    \\[Y[0,0] = \\sum_{m=0}^{2}\\sum_{n=0}^{2} K[m,n] \\cdot X[m, n]\\]
                    which computes the difference between the left column and right column of the \\(3 \\times 3\\) patch. The kernel slides across all valid positions, producing a \\(3 \\times 3\\) output (with no padding).</p></div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">Definition (Multi-Channel Convolution)</div>
                    <div class="env-body"><p>In practice, both the input and the kernel have a <strong>channel</strong> dimension. For an input \\(X \\in \\mathbb{R}^{C_{\\text{in}} \\times H \\times W}\\) and a single filter \\(K \\in \\mathbb{R}^{C_{\\text{in}} \\times k_H \\times k_W}\\), the output is
                    \\[Y[i, j] = \\sum_{c=0}^{C_{\\text{in}}-1} \\sum_{m=0}^{k_H-1} \\sum_{n=0}^{k_W-1} K[c, m, n] \\cdot X[c, i+m, j+n] + b\\]
                    where \\(b\\) is a scalar bias. A convolutional layer with \\(C_{\\text{out}}\\) filters stacks \\(C_{\\text{out}}\\) such kernels, producing an output tensor \\(Y \\in \\mathbb{R}^{C_{\\text{out}} \\times H' \\times W'}\\). The total parameter count is \\(C_{\\text{out}} \\times C_{\\text{in}} \\times k_H \\times k_W + C_{\\text{out}}\\).</p></div>
                </div>

                <div class="env-block theorem">
                    <div class="env-title">Proposition (Translation Equivariance)</div>
                    <div class="env-body"><p>Let \\(T_{(a,b)}\\) denote the operator that shifts a 2D function by \\((a,b)\\): \\((T_{(a,b)}X)[i,j] = X[i-a, j-b]\\). Then for any kernel \\(K\\),
                    \\[(T_{(a,b)}X) \\star K = T_{(a,b)}(X \\star K)\\]
                    where \\(\\star\\) denotes cross-correlation. That is, convolution commutes with translation.</p></div>
                </div>

                <div class="env-block proof">
                    <div class="env-title">Proof</div>
                    <div class="env-body"><p>Compute the left-hand side at position \\((i,j)\\):</p>
                    \\[((T_{(a,b)}X) \\star K)[i,j] = \\sum_m \\sum_n K[m,n] \\cdot (T_{(a,b)}X)[i+m, j+n] = \\sum_m \\sum_n K[m,n] \\cdot X[i+m-a, j+n-b]\\]
                    <p>The right-hand side is</p>
                    \\[T_{(a,b)}(X \\star K)[i,j] = (X \\star K)[i-a, j-b] = \\sum_m \\sum_n K[m,n] \\cdot X[(i-a)+m, (j-b)+n]\\]
                    <p>These are identical.</p>
                    <div class="qed">&#8718;</div></div>
                </div>

                <div class="viz-placeholder" data-viz="viz-conv-slide"></div>

                <div class="env-block intuition">
                    <div class="env-title">Receptive Field Growth</div>
                    <div class="env-body"><p>A single \\(3 \\times 3\\) convolutional layer gives each output unit a \\(3 \\times 3\\) receptive field. Stacking two such layers grows the receptive field to \\(5 \\times 5\\), and three layers to \\(7 \\times 7\\). In general, \\(L\\) layers of \\(k \\times k\\) kernels (stride 1) produce an effective receptive field of \\((L(k-1)+1) \\times (L(k-1)+1)\\). Deep stacks of small filters can thus capture large-scale patterns while keeping parameter counts low, a principle exploited by VGGNet.</p></div>
                </div>`,

            visualizations: [
                {
                    id: 'viz-conv-slide',
                    title: '2D Convolution: Sliding Window Animation',
                    description: 'Watch the 3\u00d73 kernel slide across the input grid. At each position, the element-wise products are summed to produce one output value. Press Play to animate, or Step to advance one position.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 700, height: 420, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;

                        // 6x6 input
                        var input = [
                            [1, 2, 0, 1, 3, 2],
                            [0, 1, 3, 2, 1, 0],
                            [2, 0, 1, 0, 2, 1],
                            [1, 3, 2, 1, 0, 3],
                            [0, 1, 0, 3, 1, 2],
                            [3, 2, 1, 0, 2, 1]
                        ];
                        // 3x3 kernel
                        var kernel = [
                            [1, 0, -1],
                            [1, 0, -1],
                            [1, 0, -1]
                        ];

                        var inRows = 6, inCols = 6, kSize = 3;
                        var outRows = inRows - kSize + 1;
                        var outCols = inCols - kSize + 1;
                        var output = [];
                        for (var r = 0; r < outRows; r++) {
                            output[r] = [];
                            for (var c = 0; c < outCols; c++) {
                                var s = 0;
                                for (var m = 0; m < kSize; m++) for (var n = 0; n < kSize; n++) s += kernel[m][n] * input[r + m][c + n];
                                output[r][c] = s;
                            }
                        }

                        var posIdx = 0;
                        var maxPos = outRows * outCols;
                        var playing = false;
                        var animTimer = null;

                        var playBtn = VizEngine.createButton(controls, 'Play', function() {
                            playing = !playing;
                            playBtn.textContent = playing ? 'Pause' : 'Play';
                            if (playing) {
                                animTimer = setInterval(function() {
                                    posIdx = (posIdx + 1) % maxPos;
                                    draw();
                                    if (posIdx === 0) { playing = false; playBtn.textContent = 'Play'; clearInterval(animTimer); }
                                }, 600);
                            } else { clearInterval(animTimer); }
                        });
                        VizEngine.createButton(controls, 'Step', function() {
                            posIdx = (posIdx + 1) % maxPos;
                            draw();
                        });
                        VizEngine.createButton(controls, 'Reset', function() {
                            posIdx = 0;
                            playing = false;
                            playBtn.textContent = 'Play';
                            if (animTimer) clearInterval(animTimer);
                            draw();
                        });

                        var cell = 44;
                        var gap = 50;

                        function drawGrid(ox, oy, data, rows, cols, label, highlightR, highlightC, highlightSize, highlightColor) {
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillStyle = '#8b949e';
                            ctx.textAlign = 'center';
                            ctx.fillText(label, ox + cols * cell / 2, oy - 10);

                            for (var r = 0; r < rows; r++) {
                                for (var c = 0; c < cols; c++) {
                                    var x = ox + c * cell, y = oy + r * cell;
                                    var isHighlight = highlightSize > 0 && r >= highlightR && r < highlightR + highlightSize && c >= highlightC && c < highlightC + highlightSize;
                                    ctx.fillStyle = isHighlight ? (highlightColor || 'rgba(88,166,255,0.25)') : '#14142e';
                                    ctx.fillRect(x, y, cell - 2, cell - 2);
                                    ctx.strokeStyle = isHighlight ? '#58a6ff' : '#30363d';
                                    ctx.lineWidth = isHighlight ? 2 : 1;
                                    ctx.strokeRect(x, y, cell - 2, cell - 2);

                                    ctx.fillStyle = isHighlight ? '#f0f6fc' : '#c9d1d9';
                                    ctx.font = '13px -apple-system,sans-serif';
                                    ctx.textAlign = 'center';
                                    ctx.textBaseline = 'middle';
                                    ctx.fillText(data[r][c], x + cell / 2 - 1, y + cell / 2 - 1);
                                }
                            }
                        }

                        function draw() {
                            viz.clear();
                            var kr = Math.floor(posIdx / outCols);
                            var kc = posIdx % outCols;

                            var inputOx = 20, inputOy = 36;
                            drawGrid(inputOx, inputOy, input, inRows, inCols, 'Input (6\u00d76)', kr, kc, kSize, 'rgba(88,166,255,0.25)');

                            var kernelOx = inputOx + inCols * cell + gap;
                            var kernelOy = inputOy + 40;
                            drawGrid(kernelOx, kernelOy, kernel, kSize, kSize, 'Kernel (3\u00d73)', -1, -1, 0);

                            var outOx = kernelOx + kSize * cell + gap;
                            var outOy = inputOy + 20;

                            // Draw output grid
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillStyle = '#8b949e';
                            ctx.textAlign = 'center';
                            ctx.fillText('Output (4\u00d74)', outOx + outCols * cell / 2, outOy - 10);

                            for (var r = 0; r < outRows; r++) {
                                for (var c = 0; c < outCols; c++) {
                                    var x = outOx + c * cell, y = outOy + r * cell;
                                    var computed = (r < kr) || (r === kr && c <= kc);
                                    var isCurrent = (r === kr && c === kc);
                                    ctx.fillStyle = isCurrent ? 'rgba(63,185,160,0.35)' : (computed ? '#1a1a40' : '#0c0c20');
                                    ctx.fillRect(x, y, cell - 2, cell - 2);
                                    ctx.strokeStyle = isCurrent ? '#3fb9a0' : (computed ? '#30363d' : '#21262d');
                                    ctx.lineWidth = isCurrent ? 2.5 : 1;
                                    ctx.strokeRect(x, y, cell - 2, cell - 2);

                                    if (computed) {
                                        ctx.fillStyle = isCurrent ? '#f0f6fc' : '#c9d1d9';
                                        ctx.font = (isCurrent ? 'bold ' : '') + '13px -apple-system,sans-serif';
                                        ctx.textAlign = 'center';
                                        ctx.textBaseline = 'middle';
                                        ctx.fillText(output[r][c], x + cell / 2 - 1, y + cell / 2 - 1);
                                    }
                                }
                            }

                            // Show the current computation
                            var compY = inputOy + inRows * cell + 28;
                            ctx.fillStyle = '#8b949e';
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('Current: Y[' + kr + ',' + kc + '] = ', 20, compY);

                            var terms = [];
                            for (var m = 0; m < kSize; m++) {
                                for (var n = 0; n < kSize; n++) {
                                    var kv = kernel[m][n];
                                    if (kv !== 0) {
                                        var iv = input[kr + m][kc + n];
                                        terms.push(kv + '\u00b7' + iv);
                                    }
                                }
                            }
                            ctx.fillStyle = '#58a6ff';
                            ctx.fillText(terms.join(' + ').replace(/\\+ -/g, '- ') + ' = ' + output[kr][kc], 140, compY);

                            // Arrow from input to output
                            ctx.strokeStyle = '#3fb9a044';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([4, 4]);
                            var ax1 = inputOx + (kc + kSize) * cell;
                            var ay1 = inputOy + (kr + 1) * cell / 2 + kr * cell / 2;
                            var ax2 = outOx + kc * cell;
                            var ay2 = outOy + kr * cell + cell / 2;
                            ctx.beginPath(); ctx.moveTo(ax1, ay1); ctx.lineTo(ax2, ay2); ctx.stroke();
                            ctx.setLineDash([]);
                        }

                        draw();

                        return {
                            stopAnimation: function() {
                                playing = false;
                                if (animTimer) clearInterval(animTimer);
                            }
                        };
                    }
                }
            ],

            exercises: [
                {
                    question: 'A single convolutional layer uses a \\(5 \\times 5\\) kernel with \\(C_{\\text{in}} = 3\\) input channels and \\(C_{\\text{out}} = 16\\) output filters. How many learnable parameters does this layer have (including biases)?',
                    hint: 'Each filter has shape \\(C_{\\text{in}} \\times k_H \\times k_W\\), and there is one bias per output filter.',
                    solution: 'Each filter: \\(3 \\times 5 \\times 5 = 75\\) weights. With 16 filters: \\(16 \\times 75 = 1200\\) weights. Adding 16 biases: \\(1200 + 16 = 1216\\) parameters total.'
                },
                {
                    question: 'Suppose you stack three \\(3 \\times 3\\) convolutional layers (stride 1, no padding). What is the effective receptive field of a single output unit in the final layer? Compare the parameter count to a single \\(7 \\times 7\\) layer (both with \\(C\\) input and \\(C\\) output channels).',
                    hint: 'Each \\(3 \\times 3\\) layer expands the receptive field by 2 in each direction. For parameters, count \\(C^2 \\times k^2\\) per layer.',
                    solution: 'Receptive field: \\(3 + 2 + 2 = 7\\), so a \\(7 \\times 7\\) effective field. Three \\(3 \\times 3\\) layers: \\(3 \\times C^2 \\times 9 = 27C^2\\) parameters. One \\(7 \\times 7\\) layer: \\(C^2 \\times 49 = 49C^2\\) parameters. The three small layers use \\(27/49 \\approx 55\\%\\) of the parameters while achieving the same receptive field and adding two more nonlinearities (deeper representation).'
                },
                {
                    question: 'Prove that translation equivariance does <em>not</em> hold for a fully connected layer in general. That is, show that there exists a weight matrix \\(W\\) and an input \\(\\mathbf{x}\\) such that \\(W \\cdot T_k[\\mathbf{x}] \\neq T_k[W \\cdot \\mathbf{x}]\\), where \\(T_k\\) is a cyclic shift.',
                    hint: 'Consider a \\(2 \\times 2\\) weight matrix and a 2D input. A cyclic shift swaps the two components.',
                    solution: 'Let \\(W = \\begin{pmatrix}1 & 0\\\\0 & 0\\end{pmatrix}\\) and \\(\\mathbf{x} = (1,0)^\\top\\). The cyclic shift gives \\(T_1[\\mathbf{x}] = (0,1)^\\top\\). Then \\(W \\cdot T_1[\\mathbf{x}] = (0,0)^\\top\\), but \\(T_1[W\\mathbf{x}] = T_1[(1,0)^\\top] = (0,1)^\\top \\neq (0,0)^\\top\\). The fully connected layer breaks equivariance because the weight matrix is not a circulant matrix.'
                }
            ]
        },

        // ===================== Section 2: Stride, Padding & Dilation =====================
        {
            id: 'ch09-sec02',
            title: 'Stride, Padding & Dilation',
            content: `<h2>Stride, Padding &amp; Dilation</h2>

                <p>The basic convolution has three hyperparameters, beyond the kernel size, that control the spatial dimensions of the output: <strong>stride</strong>, <strong>padding</strong>, and <strong>dilation</strong>. Understanding how these interact is essential for designing network architectures.</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (Stride)</div>
                    <div class="env-body"><p>The <strong>stride</strong> \\(s\\) is the number of positions the kernel moves between consecutive applications. With stride \\(s = 1\\), the kernel slides one pixel at a time (the default). With stride \\(s = 2\\), the kernel jumps two pixels, producing an output roughly half the spatial size of the input. Stride acts as a form of <em>downsampling</em>.</p></div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">Definition (Padding)</div>
                    <div class="env-body"><p><strong>Padding</strong> adds extra values (typically zeros) around the border of the input before convolution. With padding \\(p\\), the input of size \\(H \\times W\\) is expanded to \\((H + 2p) \\times (W + 2p)\\). Two common conventions:</p>
                    <ul>
                        <li><strong>Valid padding</strong> (\\(p = 0\\)): No padding. The output is strictly smaller than the input.</li>
                        <li><strong>Same padding</strong>: Choose \\(p\\) so that the output has the same spatial dimensions as the input (when stride \\(= 1\\)). For a kernel of size \\(k\\), this requires \\(p = \\lfloor k/2 \\rfloor\\).</li>
                    </ul></div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">Definition (Dilated / Atrous Convolution)</div>
                    <div class="env-body"><p>A <strong>dilated convolution</strong> with dilation rate \\(d\\) inserts \\(d - 1\\) zeros between consecutive kernel elements. The effective kernel size becomes \\(k + (k-1)(d-1) = k \\cdot d - d + 1\\), but only \\(k^2\\) parameters are used. Dilation exponentially expands the receptive field without increasing the parameter count or reducing spatial resolution.</p>
                    <p>The dilated convolution computes:</p>
                    \\[Y[i,j] = \\sum_{m=0}^{k-1}\\sum_{n=0}^{k-1} K[m,n] \\cdot X[i + m \\cdot d,\\; j + n \\cdot d]\\]</div>
                </div>

                <h3>Output Size Formula</h3>

                <div class="env-block theorem">
                    <div class="env-title">Proposition (Output Spatial Dimensions)</div>
                    <div class="env-body"><p>For an input of size \\(H_{\\text{in}}\\), kernel size \\(k\\), padding \\(p\\), stride \\(s\\), and dilation \\(d\\), the output size along one spatial dimension is
                    \\[H_{\\text{out}} = \\left\\lfloor \\frac{H_{\\text{in}} + 2p - d(k-1) - 1}{s} \\right\\rfloor + 1\\]
                    The same formula applies to the width dimension independently.</p></div>
                </div>

                <div class="env-block proof">
                    <div class="env-title">Proof</div>
                    <div class="env-body"><p>With dilation \\(d\\), the effective kernel size is \\(k_e = d(k-1) + 1\\). After padding, the effective input size is \\(H_{\\text{in}} + 2p\\). The kernel can be placed at positions \\(0, s, 2s, \\ldots\\) as long as the last kernel element \\(i + k_e - 1 \\leq H_{\\text{in}} + 2p - 1\\). The number of valid positions is
                    \\[\\left\\lfloor \\frac{(H_{\\text{in}} + 2p) - k_e}{s} \\right\\rfloor + 1 = \\left\\lfloor \\frac{H_{\\text{in}} + 2p - d(k-1) - 1}{s} \\right\\rfloor + 1\\]</p>
                    <div class="qed">&#8718;</div></div>
                </div>

                <div class="env-block example">
                    <div class="env-title">Example (Common Configurations)</div>
                    <div class="env-body">
                    <p><strong>VGG-style:</strong> \\(k=3, p=1, s=1, d=1\\). Output size: \\(\\lfloor(H + 2 - 3)/1\\rfloor + 1 = H\\). Spatial dimensions are preserved.</p>
                    <p><strong>Strided downsampling:</strong> \\(k=3, p=1, s=2, d=1\\). Output: \\(\\lfloor(H + 2 - 3)/2\\rfloor + 1 = \\lfloor(H-1)/2\\rfloor + 1\\). For \\(H = 32\\): output is 16. This halves the spatial size, replacing pooling in modern architectures like ResNet.</p>
                    <p><strong>Dilated convolution:</strong> \\(k=3, p=2, s=1, d=2\\). Effective kernel: \\(2(3-1)+1 = 5\\). Output: \\(\\lfloor(H + 4 - 5)/1\\rfloor + 1 = H\\). Same spatial size with a \\(5 \\times 5\\) receptive field using only 9 parameters.</p>
                    </div>
                </div>

                <div class="viz-placeholder" data-viz="viz-stride-pad"></div>

                <div class="env-block warning">
                    <div class="env-title">Warning (Information Loss at Boundaries)</div>
                    <div class="env-body"><p>With valid padding (\\(p=0\\)), border pixels contribute to fewer output positions than center pixels. Over many layers, this creates an implicit spatial bias where boundary information is progressively lost. Same padding mitigates this but introduces zeros that can affect the statistics of boundary activations. Some architectures use reflection or replication padding as alternatives.</p></div>
                </div>

                <div class="env-block intuition">
                    <div class="env-title">When to Use Dilation</div>
                    <div class="env-body"><p>Dilated convolutions are especially useful in <em>dense prediction</em> tasks (semantic segmentation, audio generation) where you need a large receptive field without losing spatial resolution. DeepLab (Chen et al., 2017) uses "atrous spatial pyramid pooling" with multiple dilation rates in parallel to capture multi-scale context. WaveNet (van den Oord et al., 2016) stacks exponentially increasing dilation rates (1, 2, 4, 8, ...) to achieve receptive fields spanning thousands of time steps with logarithmic depth.</p></div>
                </div>`,

            visualizations: [
                {
                    id: 'viz-stride-pad',
                    title: 'Interactive Stride, Padding & Dilation Demo',
                    description: 'Adjust the sliders to see how stride, padding, and dilation affect the output dimensions. The blue region shows the kernel placement; gray cells are padding.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 700, height: 440, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;

                        var inSize = 7;
                        var kSize = 3;
                        var stride = 1;
                        var padding = 0;
                        var dilation = 1;

                        VizEngine.createSlider(controls, 'Input', 5, 10, inSize, 1, function(v) { inSize = Math.round(v); draw(); });
                        VizEngine.createSlider(controls, 'Kernel', 2, 5, kSize, 1, function(v) { kSize = Math.round(v); draw(); });
                        VizEngine.createSlider(controls, 'Stride', 1, 4, stride, 1, function(v) { stride = Math.round(v); draw(); });
                        VizEngine.createSlider(controls, 'Pad', 0, 4, padding, 1, function(v) { padding = Math.round(v); draw(); });
                        VizEngine.createSlider(controls, 'Dilation', 1, 4, dilation, 1, function(v) { dilation = Math.round(v); draw(); });

                        function computeOutSize(H, k, p, s, d) {
                            var effK = d * (k - 1) + 1;
                            var val = Math.floor((H + 2 * p - effK) / s) + 1;
                            return Math.max(val, 0);
                        }

                        function draw() {
                            viz.clear();

                            var outH = computeOutSize(inSize, kSize, padding, stride, dilation);
                            var effK = dilation * (kSize - 1) + 1;
                            var padded = inSize + 2 * padding;

                            // Auto-size cells
                            var maxGridDim = Math.max(padded, outH, 1);
                            var cell = Math.min(38, Math.floor(280 / maxGridDim));
                            cell = Math.max(cell, 18);

                            // Draw padded input grid
                            var ox = 20, oy = 50;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillStyle = '#8b949e';
                            ctx.textAlign = 'center';
                            ctx.fillText('Padded Input (' + padded + '\u00d7' + padded + ')', ox + padded * cell / 2, oy - 12);

                            for (var r = 0; r < padded; r++) {
                                for (var c = 0; c < padded; c++) {
                                    var x = ox + c * cell, y = oy + r * cell;
                                    var isPad = r < padding || r >= padding + inSize || c < padding || c >= padding + inSize;
                                    ctx.fillStyle = isPad ? '#1a1a40' : '#14142e';
                                    ctx.fillRect(x, y, cell - 1, cell - 1);
                                    ctx.strokeStyle = isPad ? '#333366' : '#30363d';
                                    ctx.lineWidth = 1;
                                    ctx.strokeRect(x, y, cell - 1, cell - 1);

                                    if (isPad && cell >= 22) {
                                        ctx.fillStyle = '#555577';
                                        ctx.font = '10px -apple-system,sans-serif';
                                        ctx.textBaseline = 'middle';
                                        ctx.fillText('0', x + cell / 2, y + cell / 2);
                                    }
                                }
                            }

                            // Highlight first kernel position with dilation
                            if (outH > 0 && effK <= padded) {
                                for (var m = 0; m < kSize; m++) {
                                    for (var n = 0; n < kSize; n++) {
                                        var gr = m * dilation;
                                        var gc = n * dilation;
                                        if (gr < padded && gc < padded) {
                                            var x2 = ox + gc * cell, y2 = oy + gr * cell;
                                            ctx.fillStyle = 'rgba(88,166,255,0.3)';
                                            ctx.fillRect(x2, y2, cell - 1, cell - 1);
                                            ctx.strokeStyle = '#58a6ff';
                                            ctx.lineWidth = 2;
                                            ctx.strokeRect(x2, y2, cell - 1, cell - 1);
                                        }
                                    }
                                }
                                // Outline the effective kernel bounding box
                                ctx.strokeStyle = '#58a6ff66';
                                ctx.lineWidth = 1;
                                ctx.setLineDash([3, 3]);
                                ctx.strokeRect(ox, oy, effK * cell - 1, effK * cell - 1);
                                ctx.setLineDash([]);
                            }

                            // Draw output grid
                            var outOx = ox + padded * cell + 60;
                            var outOy = oy;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillStyle = '#8b949e';
                            ctx.textAlign = 'center';
                            ctx.fillText('Output (' + outH + '\u00d7' + outH + ')', outOx + Math.max(outH, 1) * cell / 2, outOy - 12);

                            if (outH > 0) {
                                for (var r2 = 0; r2 < outH; r2++) {
                                    for (var c2 = 0; c2 < outH; c2++) {
                                        var xx = outOx + c2 * cell, yy = outOy + r2 * cell;
                                        ctx.fillStyle = (r2 === 0 && c2 === 0) ? 'rgba(63,185,160,0.3)' : '#14142e';
                                        ctx.fillRect(xx, yy, cell - 1, cell - 1);
                                        ctx.strokeStyle = (r2 === 0 && c2 === 0) ? '#3fb9a0' : '#30363d';
                                        ctx.lineWidth = (r2 === 0 && c2 === 0) ? 2 : 1;
                                        ctx.strokeRect(xx, yy, cell - 1, cell - 1);
                                    }
                                }
                            } else {
                                ctx.fillStyle = '#f85149';
                                ctx.font = '13px -apple-system,sans-serif';
                                ctx.fillText('Invalid config', outOx + 50, outOy + 30);
                                ctx.fillText('(kernel larger', outOx + 50, outOy + 50);
                                ctx.fillText('than input)', outOx + 50, outOy + 70);
                            }

                            // Formula display
                            var formY = oy + Math.max(padded, outH) * cell + 30;
                            ctx.fillStyle = '#c9d1d9';
                            ctx.font = '13px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('H_out = floor((' + inSize + ' + 2\u00b7' + padding + ' - ' + dilation + '\u00b7(' + kSize + '-1) - 1) / ' + stride + ') + 1', 20, formY);
                            ctx.fillStyle = '#3fb9a0';
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.fillText('= floor((' + (inSize + 2 * padding) + ' - ' + effK + ') / ' + stride + ') + 1 = ' + outH, 20, formY + 22);

                            if (dilation > 1) {
                                ctx.fillStyle = '#58a6ff';
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.fillText('Effective kernel: ' + effK + '\u00d7' + effK + '  (dilation=' + dilation + ', actual params=' + kSize + '\u00d7' + kSize + '=' + (kSize * kSize) + ')', 20, formY + 46);
                            }
                        }

                        draw();
                        return viz;
                    }
                }
            ],

            exercises: [
                {
                    question: 'An input of size \\(32 \\times 32\\) is convolved with a \\(5 \\times 5\\) kernel using stride 2 and padding 2. What is the output size?',
                    hint: 'Apply the formula: \\(H_{\\text{out}} = \\lfloor (H_{\\text{in}} + 2p - k) / s \\rfloor + 1\\).',
                    solution: '\\(H_{\\text{out}} = \\lfloor (32 + 4 - 5) / 2 \\rfloor + 1 = \\lfloor 31/2 \\rfloor + 1 = 15 + 1 = 16\\). The output is \\(16 \\times 16\\).'
                },
                {
                    question: 'What padding \\(p\\) is needed to achieve "same" output size (\\(H_{\\text{out}} = H_{\\text{in}}\\)) with a \\(k \\times k\\) kernel, stride 1, and dilation \\(d\\)?',
                    hint: 'Set \\(H_{\\text{out}} = H_{\\text{in}}\\) in the formula and solve for \\(p\\), using \\(s = 1\\).',
                    solution: 'Setting \\(H_{\\text{out}} = H_{\\text{in}}\\): \\(H = (H + 2p - d(k-1) - 1)/1 + 1\\), so \\(H = H + 2p - d(k-1)\\), giving \\(p = d(k-1)/2\\). For this to be an integer, \\(k\\) must be odd (the standard convention). With \\(d=1\\): \\(p = (k-1)/2\\). For example, \\(k=3 \\Rightarrow p=1\\); \\(k=5 \\Rightarrow p=2\\).'
                },
                {
                    question: 'A WaveNet-style architecture stacks 1D dilated convolutions with \\(k=2\\) and dilation rates \\(d = 1, 2, 4, 8, 16\\). What is the total receptive field? How many layers would you need with \\(k=2, d=1\\) (no dilation) to achieve the same receptive field?',
                    hint: 'For dilation \\(d\\) and kernel \\(k\\), each layer adds \\(d \\cdot (k-1)\\) to the receptive field. The receptive field of stacked layers is \\(1 + \\sum_l d_l(k-1)\\).',
                    solution: 'Receptive field: \\(1 + (1 + 2 + 4 + 8 + 16)(2 - 1) = 1 + 31 = 32\\). With no dilation (\\(d=1, k=2\\)), each layer adds 1 to the receptive field, so you would need 31 layers to reach a receptive field of 32. Exponential dilation achieves the same with only 5 layers, a logarithmic-depth construction.'
                }
            ]
        },

        // ===================== Section 3: Pooling Operations =====================
        {
            id: 'ch09-sec03',
            title: 'Pooling Operations',
            content: `<h2>Pooling Operations</h2>

                <p>Pooling layers reduce the spatial dimensions of feature maps, providing a form of translation invariance and reducing computation for subsequent layers. Unlike convolutional layers, pooling has <strong>no learnable parameters</strong>.</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (Pooling)</div>
                    <div class="env-body"><p>A <strong>pooling</strong> operation partitions each channel of the input feature map into non-overlapping (or overlapping) rectangular regions and computes a summary statistic for each region. Given a pooling window of size \\(k \\times k\\) and stride \\(s\\), the output at position \\((i,j)\\) for channel \\(c\\) is
                    \\[Y[c, i, j] = \\text{pool}\\bigl(\\{X[c,\\, i \\cdot s + m,\\, j \\cdot s + n] : 0 \\le m, n &lt; k\\}\\bigr)\\]
                    where \\(\\text{pool}(\\cdot)\\) is the aggregation function.</p></div>
                </div>

                <h3>Types of Pooling</h3>

                <div class="env-block definition">
                    <div class="env-title">Definition (Max Pooling)</div>
                    <div class="env-body"><p><strong>Max pooling</strong> selects the maximum value in each window:
                    \\[Y[c, i, j] = \\max_{0 \\le m,n &lt; k} X[c,\\, i \\cdot s + m,\\, j \\cdot s + n]\\]
                    Max pooling retains the strongest activation in each region, which is useful for detecting whether a feature is present <em>somewhere</em> in the neighborhood (approximate translation invariance).</p></div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">Definition (Average Pooling)</div>
                    <div class="env-body"><p><strong>Average pooling</strong> computes the mean value in each window:
                    \\[Y[c, i, j] = \\frac{1}{k^2} \\sum_{m=0}^{k-1}\\sum_{n=0}^{k-1} X[c,\\, i \\cdot s + m,\\, j \\cdot s + n]\\]
                    Average pooling is a linear, smoothing operation. It distributes the gradient equally across all elements in the window during backpropagation.</p></div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">Definition (Global Average Pooling)</div>
                    <div class="env-body"><p><strong>Global average pooling (GAP)</strong> averages each entire channel into a single scalar:
                    \\[Y[c] = \\frac{1}{H \\times W} \\sum_{i=0}^{H-1}\\sum_{j=0}^{W-1} X[c, i, j]\\]
                    This maps a tensor \\(\\in \\mathbb{R}^{C \\times H \\times W}\\) to a vector \\(\\in \\mathbb{R}^C\\), eliminating all spatial dimensions. GAP was introduced by Lin et al. (2013, Network in Network) and is now the standard way to transition from convolutional features to the classification head, replacing fully connected layers and dramatically reducing parameters.</p></div>
                </div>

                <h3>Pooling and Invariance</h3>

                <p>Max pooling introduces a degree of <em>translation invariance</em> (not just equivariance). If a feature activation shifts by one pixel within a \\(2 \\times 2\\) max-pooling window, the pooled output remains unchanged, as long as the maximum stays the maximum. This invariance is approximate and local; large shifts will change the output.</p>

                <div class="env-block remark">
                    <div class="env-title">Remark (Pooling Gradients)</div>
                    <div class="env-body"><p>During backpropagation through max pooling, the gradient is routed entirely to the element that achieved the maximum (the "argmax"), and all other elements receive zero gradient. This can be interpreted as a form of hard attention. For average pooling, the gradient is divided equally among all elements in the window: \\(\\partial L / \\partial X[c, i \\cdot s + m, j \\cdot s + n] = \\frac{1}{k^2} \\cdot \\partial L / \\partial Y[c,i,j]\\).</p></div>
                </div>

                <div class="env-block example">
                    <div class="env-title">Example (2&times;2 Max Pool and Average Pool)</div>
                    <div class="env-body"><p>Consider a \\(4 \\times 4\\) feature map:
                    \\[X = \\begin{pmatrix} 1 & 3 & 2 & 4 \\\\ 5 & 6 & 1 & 2 \\\\ 7 & 2 & 0 & 3 \\\\ 1 & 4 & 5 & 6 \\end{pmatrix}\\]
                    With \\(2 \\times 2\\) pooling (stride 2):</p>
                    <p><strong>Max pool:</strong> \\(\\begin{pmatrix} 6 & 4 \\\\ 7 & 6 \\end{pmatrix}\\) &nbsp;&nbsp;&nbsp; <strong>Avg pool:</strong> \\(\\begin{pmatrix} 3.75 & 2.25 \\\\ 3.5 & 3.5 \\end{pmatrix}\\)</p></div>
                </div>

                <div class="viz-placeholder" data-viz="viz-pooling"></div>

                <div class="env-block intuition">
                    <div class="env-title">Modern Trend: Strided Convolution over Pooling</div>
                    <div class="env-body"><p>Many modern architectures (ResNet, EfficientNet) replace traditional pooling layers with strided convolutions for downsampling. The argument is that the network should <em>learn</em> how to downsample rather than using a fixed operation. Strided convolutions achieve comparable or better accuracy while giving the network more flexibility. However, GAP remains universally used at the final stage.</p></div>
                </div>`,

            visualizations: [
                {
                    id: 'viz-pooling',
                    title: 'Max Pool vs. Average Pool Comparison',
                    description: 'A 6\u00d76 feature map is pooled using both max and average pooling. Colored blocks show which input regions map to which output cells. The highlighted window shows the current pooling region.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 700, height: 400, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;

                        var input = [
                            [1, 3, 2, 4, 0, 5],
                            [5, 6, 1, 2, 3, 1],
                            [7, 2, 0, 3, 8, 2],
                            [1, 4, 5, 6, 1, 4],
                            [3, 0, 2, 1, 7, 3],
                            [2, 5, 4, 3, 6, 9]
                        ];
                        var poolSize = 2;
                        var inN = 6;
                        var outN = inN / poolSize;
                        var highlightR = 0, highlightC = 0;

                        var regionColors = [
                            'rgba(88,166,255,0.18)', 'rgba(63,185,160,0.18)', 'rgba(240,136,62,0.18)',
                            'rgba(188,140,255,0.18)', 'rgba(63,185,96,0.18)', 'rgba(247,120,186,0.18)',
                            'rgba(210,153,34,0.18)', 'rgba(248,81,73,0.18)', 'rgba(88,166,255,0.25)'
                        ];

                        var cell = 46;

                        function computePools() {
                            var maxPool = [], avgPool = [];
                            for (var r = 0; r < outN; r++) {
                                maxPool[r] = []; avgPool[r] = [];
                                for (var c = 0; c < outN; c++) {
                                    var mx = -Infinity, sm = 0;
                                    for (var m = 0; m < poolSize; m++) for (var n = 0; n < poolSize; n++) {
                                        var v = input[r * poolSize + m][c * poolSize + n];
                                        mx = Math.max(mx, v); sm += v;
                                    }
                                    maxPool[r][c] = mx;
                                    avgPool[r][c] = sm / (poolSize * poolSize);
                                }
                            }
                            return { max: maxPool, avg: avgPool };
                        }

                        function drawLabeledGrid(ox, oy, data, rows, cols, label, cellSz, hlRow, hlCol, hlSize) {
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillStyle = '#8b949e';
                            ctx.textAlign = 'center';
                            ctx.fillText(label, ox + cols * cellSz / 2, oy - 10);

                            for (var r = 0; r < rows; r++) {
                                for (var c = 0; c < cols; c++) {
                                    var x = ox + c * cellSz, y = oy + r * cellSz;

                                    // Region coloring for input
                                    var regionIdx = -1;
                                    if (hlSize > 0 && poolSize > 0) {
                                        regionIdx = Math.floor(r / poolSize) * outN + Math.floor(c / poolSize);
                                    }
                                    var isHl = hlSize > 0 && r >= hlRow * poolSize && r < (hlRow + 1) * poolSize && c >= hlCol * poolSize && c < (hlCol + 1) * poolSize;

                                    ctx.fillStyle = isHl ? 'rgba(88,166,255,0.3)' : (regionIdx >= 0 ? regionColors[regionIdx % regionColors.length] : '#14142e');
                                    ctx.fillRect(x, y, cellSz - 2, cellSz - 2);
                                    ctx.strokeStyle = isHl ? '#58a6ff' : '#30363d';
                                    ctx.lineWidth = isHl ? 2 : 1;
                                    ctx.strokeRect(x, y, cellSz - 2, cellSz - 2);

                                    ctx.fillStyle = isHl ? '#f0f6fc' : '#c9d1d9';
                                    ctx.font = '13px -apple-system,sans-serif';
                                    ctx.textAlign = 'center';
                                    ctx.textBaseline = 'middle';
                                    var val = data[r][c];
                                    ctx.fillText(typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(1)) : val, x + cellSz / 2 - 1, y + cellSz / 2 - 1);
                                }
                            }
                        }

                        function draw() {
                            viz.clear();
                            var pools = computePools();
                            var ox = 16, oy = 42;

                            drawLabeledGrid(ox, oy, input, inN, inN, 'Input (6\u00d76)', cell, highlightR, highlightC, poolSize);

                            var outCell = cell;
                            var gap = 40;
                            var maxOx = ox + inN * cell + gap;
                            var avgOx = maxOx + outN * outCell + gap + 20;

                            // Max pool output
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillStyle = '#f0883e';
                            ctx.textAlign = 'center';
                            ctx.fillText('Max Pool', maxOx + outN * outCell / 2, oy - 10);

                            for (var r = 0; r < outN; r++) {
                                for (var c = 0; c < outN; c++) {
                                    var x = maxOx + c * outCell, y = oy + r * outCell;
                                    var isCur = r === highlightR && c === highlightC;
                                    ctx.fillStyle = isCur ? 'rgba(240,136,62,0.3)' : '#14142e';
                                    ctx.fillRect(x, y, outCell - 2, outCell - 2);
                                    ctx.strokeStyle = isCur ? '#f0883e' : '#30363d';
                                    ctx.lineWidth = isCur ? 2 : 1;
                                    ctx.strokeRect(x, y, outCell - 2, outCell - 2);
                                    ctx.fillStyle = isCur ? '#f0f6fc' : '#c9d1d9';
                                    ctx.font = '13px -apple-system,sans-serif';
                                    ctx.textAlign = 'center';
                                    ctx.textBaseline = 'middle';
                                    ctx.fillText(pools.max[r][c], x + outCell / 2 - 1, y + outCell / 2 - 1);
                                }
                            }

                            // Avg pool output
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillStyle = '#bc8cff';
                            ctx.textAlign = 'center';
                            ctx.fillText('Avg Pool', avgOx + outN * outCell / 2, oy - 10);

                            for (var r2 = 0; r2 < outN; r2++) {
                                for (var c2 = 0; c2 < outN; c2++) {
                                    var xx = avgOx + c2 * outCell, yy = oy + r2 * outCell;
                                    var isCur2 = r2 === highlightR && c2 === highlightC;
                                    ctx.fillStyle = isCur2 ? 'rgba(188,140,255,0.3)' : '#14142e';
                                    ctx.fillRect(xx, yy, outCell - 2, outCell - 2);
                                    ctx.strokeStyle = isCur2 ? '#bc8cff' : '#30363d';
                                    ctx.lineWidth = isCur2 ? 2 : 1;
                                    ctx.strokeRect(xx, yy, outCell - 2, outCell - 2);
                                    ctx.fillStyle = isCur2 ? '#f0f6fc' : '#c9d1d9';
                                    ctx.font = '13px -apple-system,sans-serif';
                                    ctx.textAlign = 'center';
                                    ctx.textBaseline = 'middle';
                                    ctx.fillText(pools.avg[r2][c2].toFixed(1), xx + outCell / 2 - 1, yy + outCell / 2 - 1);
                                }
                            }

                            // Info line
                            var infoY = oy + inN * cell + 24;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillStyle = '#8b949e';
                            var vals = [];
                            for (var mm = 0; mm < poolSize; mm++) for (var nn = 0; nn < poolSize; nn++) vals.push(input[highlightR * poolSize + mm][highlightC * poolSize + nn]);
                            ctx.fillText('Region [' + highlightR + ',' + highlightC + ']: {' + vals.join(', ') + '}', 16, infoY);
                            ctx.fillStyle = '#f0883e';
                            ctx.fillText('max = ' + pools.max[highlightR][highlightC], 16, infoY + 20);
                            ctx.fillStyle = '#bc8cff';
                            ctx.fillText('avg = ' + pools.avg[highlightR][highlightC].toFixed(2), 140, infoY + 20);
                        }

                        // Click to select region
                        viz.canvas.addEventListener('click', function(e) {
                            var rect = viz.canvas.getBoundingClientRect();
                            var mx = e.clientX - rect.left;
                            var my = e.clientY - rect.top;
                            var ox = 16, oy = 42;
                            var gr = Math.floor((my - oy) / cell);
                            var gc = Math.floor((mx - ox) / cell);
                            if (gr >= 0 && gr < inN && gc >= 0 && gc < inN) {
                                highlightR = Math.floor(gr / poolSize);
                                highlightC = Math.floor(gc / poolSize);
                                if (highlightR >= outN) highlightR = outN - 1;
                                if (highlightC >= outN) highlightC = outN - 1;
                                draw();
                            }
                        });

                        draw();
                        return viz;
                    }
                }
            ],

            exercises: [
                {
                    question: 'A feature map of size \\(C \\times 14 \\times 14\\) undergoes \\(2 \\times 2\\) max pooling with stride 2. What is the output size? How many parameters does this pooling layer have?',
                    hint: 'Pooling operates on each channel independently and has no learnable parameters.',
                    solution: 'Output: \\(C \\times 7 \\times 7\\). Pooling has <strong>zero</strong> learnable parameters; it is a fixed, non-parametric operation. Note that pooling is applied per-channel: the channel dimension \\(C\\) is unchanged.'
                },
                {
                    question: 'Explain why max pooling is not strictly translation invariant. Construct a 1D example with pool size 2 and stride 2 where shifting the input by 1 position changes the max-pooled output.',
                    hint: 'Consider an input where the maximum of each pool window changes after a shift.',
                    solution: 'Consider \\(x = [1, 5, 2, 3]\\). Max pool with size 2, stride 2: \\([\\max(1,5), \\max(2,3)] = [5, 3]\\). Shift by 1 (with zero padding at left): \\(x\' = [0, 1, 5, 2]\\). Max pool: \\([\\max(0,1), \\max(5,2)] = [1, 5]\\). The outputs \\([5, 3]\\) and \\([1, 5]\\) are different, so max pooling is not strictly translation invariant. It only provides invariance to shifts <em>within</em> each pooling window.'
                },
                {
                    question: 'In the gradient computation for max pooling, only the "winning" element receives a gradient. What potential problem could this cause during training, and how does average pooling address it?',
                    hint: 'Think about what happens to elements that are never the maximum in any pooling window.',
                    solution: 'Elements that are never the local maximum receive zero gradient through the max-pool layer, which can slow their updates. In pathological cases, "dead" spatial regions may persist. Average pooling distributes the gradient equally to all elements in the window (each gets \\(1/k^2\\) of the output gradient), ensuring every element receives a learning signal. In practice, this is rarely a serious problem for max pooling because the convolution parameters before pooling receive gradients anyway, but it is a reason why average pooling is sometimes preferred in certain architectures (e.g., global average pooling in ResNet).'
                }
            ]
        },

        // ===================== Section 4: Filter Effects =====================
        {
            id: 'ch09-sec04',
            title: 'Filter Effects',
            content: `<h2>Filter Effects</h2>

                <p>Before the era of learned features, image processing relied on hand-crafted kernels for tasks like edge detection, blurring, and sharpening. Understanding these classical filters builds intuition for what convolutional networks learn automatically.</p>

                <h3>Classical 3&times;3 Kernels</h3>

                <div class="env-block definition">
                    <div class="env-title">Edge Detection Kernels</div>
                    <div class="env-body"><p>Edge detectors compute spatial gradients, highlighting locations where pixel intensity changes sharply.</p>
                    <p><strong>Horizontal Sobel:</strong> Detects horizontal edges (vertical gradients)
                    \\[K_h = \\begin{pmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 2 & 1 \\end{pmatrix}\\]</p>
                    <p><strong>Vertical Sobel:</strong> Detects vertical edges (horizontal gradients)
                    \\[K_v = \\begin{pmatrix} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \\end{pmatrix}\\]</p>
                    <p>The gradient magnitude \\(\\sqrt{(K_h * X)^2 + (K_v * X)^2}\\) gives the overall edge strength at each pixel.</p></div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">Blur (Box Filter)</div>
                    <div class="env-body"><p>The <strong>box blur</strong> averages all values in the neighborhood, smoothing out noise:
                    \\[K_{\\text{blur}} = \\frac{1}{9}\\begin{pmatrix} 1 & 1 & 1 \\\\ 1 & 1 & 1 \\\\ 1 & 1 & 1 \\end{pmatrix}\\]
                    A Gaussian blur uses weights decaying with distance from the center, producing smoother results:
                    \\[K_{\\text{gauss}} \\approx \\frac{1}{16}\\begin{pmatrix} 1 & 2 & 1 \\\\ 2 & 4 & 2 \\\\ 1 & 2 & 1 \\end{pmatrix}\\]</p></div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">Sharpen Filter</div>
                    <div class="env-body"><p>The <strong>sharpen</strong> kernel enhances edges by amplifying the center pixel relative to its neighbors:
                    \\[K_{\\text{sharp}} = \\begin{pmatrix} 0 & -1 & 0 \\\\ -1 & 5 & -1 \\\\ 0 & -1 & 0 \\end{pmatrix}\\]
                    This can be decomposed as \\(K_{\\text{sharp}} = I + \\alpha \\cdot K_{\\text{laplacian}}\\) where \\(I\\) is the identity kernel (passing the image through unchanged) and \\(K_{\\text{laplacian}}\\) is a discrete Laplacian that detects second-order spatial derivatives.</p></div>
                </div>

                <h3>What CNNs Learn</h3>

                <p>Empirical studies (Zeiler &amp; Fergus, 2014) show that the first convolutional layer of a trained CNN learns kernels strikingly similar to Gabor filters and edge detectors. Deeper layers learn progressively more abstract features: textures in layer 2, object parts in layer 3, and object-level concepts in layers 4-5. The remarkable finding is that training on natural images reliably recovers these classical image processing operations as a byproduct of minimizing classification loss.</p>

                <div class="env-block remark">
                    <div class="env-title">Remark (Kernel Decomposition)</div>
                    <div class="env-body"><p>Any filter can be understood through its frequency response. A blur kernel is a <em>low-pass filter</em> that attenuates high frequencies (edges, noise). An edge detector is a <em>high-pass filter</em> that suppresses smooth regions and amplifies transitions. The sharpen kernel boosts high frequencies relative to the original signal. CNNs learn a bank of filters spanning different frequency bands and orientations, collectively covering the information needed for the task.</p></div>
                </div>

                <div class="viz-placeholder" data-viz="viz-filters"></div>

                <div class="env-block intuition">
                    <div class="env-title">From Filters to Features</div>
                    <div class="env-body"><p>A single kernel detects a single pattern. A convolutional layer with \\(C_{\\text{out}}\\) kernels produces \\(C_{\\text{out}}\\) feature maps, each responding to a different pattern. By stacking layers, the network composes these detectors: layer 1 detects edges, layer 2 combines edges into corners and textures, layer 3 combines those into object parts, and so on. This hierarchical composition is the key insight behind the success of deep convolutional networks.</p></div>
                </div>`,

            visualizations: [
                {
                    id: 'viz-filters',
                    title: 'Filter Effects on a Pattern',
                    description: 'Select a filter to see its effect on the input pattern. The input contains a checkerboard with a bright square, showing how different filters respond to edges and flat regions.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 700, height: 420, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;

                        var N = 9;
                        // Create a pattern with edges and flat regions
                        var inputData = [
                            [0, 0, 0, 8, 8, 8, 0, 0, 0],
                            [0, 0, 0, 8, 8, 8, 0, 0, 0],
                            [0, 0, 0, 8, 8, 8, 0, 0, 0],
                            [8, 8, 8, 4, 4, 4, 8, 8, 8],
                            [8, 8, 8, 4, 4, 4, 8, 8, 8],
                            [8, 8, 8, 4, 4, 4, 8, 8, 8],
                            [0, 0, 0, 8, 8, 8, 0, 0, 0],
                            [0, 0, 0, 8, 8, 8, 0, 0, 0],
                            [0, 0, 0, 8, 8, 8, 0, 0, 0]
                        ];

                        var kernels = {
                            'Identity': { k: [[0,0,0],[0,1,0],[0,0,0]], color: '#8b949e' },
                            'Sobel H': { k: [[-1,-2,-1],[0,0,0],[1,2,1]], color: '#58a6ff' },
                            'Sobel V': { k: [[-1,0,1],[-2,0,2],[-1,0,1]], color: '#3fb9a0' },
                            'Blur':    { k: [[1,1,1],[1,1,1],[1,1,1]], norm: 9, color: '#bc8cff' },
                            'Sharpen': { k: [[0,-1,0],[-1,5,-1],[0,-1,0]], color: '#f0883e' },
                            'Laplacian': { k: [[0,1,0],[1,-4,1],[0,1,0]], color: '#f85149' }
                        };
                        var currentFilter = 'Identity';

                        // Create filter buttons
                        Object.keys(kernels).forEach(function(name) {
                            var btn = VizEngine.createButton(controls, name, function() {
                                currentFilter = name;
                                draw();
                            });
                            btn.style.fontSize = '0.72rem';
                            btn.style.padding = '3px 8px';
                        });

                        function convolve(inp, kern, norm) {
                            var outN = N - 2;
                            var out = [];
                            for (var r = 0; r < outN; r++) {
                                out[r] = [];
                                for (var c = 0; c < outN; c++) {
                                    var s = 0;
                                    for (var m = 0; m < 3; m++) for (var n = 0; n < 3; n++) s += kern[m][n] * inp[r + m][c + n];
                                    if (norm) s /= norm;
                                    out[r][c] = s;
                                }
                            }
                            return out;
                        }

                        function valToColor(v, minV, maxV) {
                            var range = maxV - minV || 1;
                            var t = (v - minV) / range;
                            t = Math.max(0, Math.min(1, t));
                            var r = Math.round(t * 220 + 20);
                            var g = Math.round(t * 220 + 20);
                            var b = Math.round(t * 255 + 30);
                            return 'rgb(' + r + ',' + g + ',' + b + ')';
                        }

                        function drawHeatGrid(ox, oy, data, rows, cols, label, cellSz, labelColor) {
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillStyle = labelColor || '#8b949e';
                            ctx.textAlign = 'center';
                            ctx.fillText(label, ox + cols * cellSz / 2, oy - 10);

                            var minV = Infinity, maxV = -Infinity;
                            for (var rr = 0; rr < rows; rr++) for (var cc = 0; cc < cols; cc++) {
                                minV = Math.min(minV, data[rr][cc]);
                                maxV = Math.max(maxV, data[rr][cc]);
                            }

                            for (var r = 0; r < rows; r++) {
                                for (var c = 0; c < cols; c++) {
                                    var x = ox + c * cellSz, y = oy + r * cellSz;
                                    ctx.fillStyle = valToColor(data[r][c], minV, maxV);
                                    ctx.fillRect(x, y, cellSz - 1, cellSz - 1);
                                    ctx.strokeStyle = '#30363d';
                                    ctx.lineWidth = 0.5;
                                    ctx.strokeRect(x, y, cellSz - 1, cellSz - 1);

                                    if (cellSz >= 30) {
                                        var v = data[r][c];
                                        var bright = (v - minV) / (maxV - minV || 1) > 0.5;
                                        ctx.fillStyle = bright ? '#0c0c20' : '#f0f6fc';
                                        ctx.font = '11px -apple-system,sans-serif';
                                        ctx.textAlign = 'center';
                                        ctx.textBaseline = 'middle';
                                        ctx.fillText(Number.isInteger(v) ? v : v.toFixed(1), x + cellSz / 2, y + cellSz / 2);
                                    }
                                }
                            }
                        }

                        function draw() {
                            viz.clear();
                            var filt = kernels[currentFilter];
                            var output = convolve(inputData, filt.k, filt.norm);
                            var inCell = 36;
                            var outCell = 36;

                            var ox = 20, oy = 50;
                            drawHeatGrid(ox, oy, inputData, N, N, 'Input (9\u00d79)', inCell, '#8b949e');

                            // Draw kernel
                            var kOx = ox + N * inCell + 30;
                            var kOy = oy + 60;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillStyle = filt.color;
                            ctx.textAlign = 'center';
                            ctx.fillText(currentFilter + ' Kernel', kOx + 1.5 * 48, kOy - 12);

                            var kCell = 48;
                            for (var m = 0; m < 3; m++) {
                                for (var nn = 0; nn < 3; nn++) {
                                    var kx = kOx + nn * kCell, ky = kOy + m * kCell;
                                    var kv = filt.k[m][nn];
                                    var denom = filt.norm || 1;
                                    ctx.fillStyle = kv > 0 ? 'rgba(63,185,160,0.2)' : (kv < 0 ? 'rgba(248,81,73,0.2)' : '#14142e');
                                    ctx.fillRect(kx, ky, kCell - 2, kCell - 2);
                                    ctx.strokeStyle = filt.color;
                                    ctx.lineWidth = 1.5;
                                    ctx.strokeRect(kx, ky, kCell - 2, kCell - 2);

                                    ctx.fillStyle = '#f0f6fc';
                                    ctx.font = '14px -apple-system,sans-serif';
                                    ctx.textAlign = 'center';
                                    ctx.textBaseline = 'middle';
                                    var dispV = filt.norm ? (kv / filt.norm).toFixed(2) : kv;
                                    ctx.fillText(dispV, kx + kCell / 2 - 1, ky + kCell / 2 - 1);
                                }
                            }

                            // Draw output
                            var outOx = kOx + 3 * kCell + 30;
                            var outOy = oy;
                            drawHeatGrid(outOx, outOy, output, N - 2, N - 2, 'Output (7\u00d77)', outCell, filt.color);

                            // Arrow
                            ctx.strokeStyle = filt.color + '66';
                            ctx.lineWidth = 1.5;
                            ctx.setLineDash([4, 4]);
                            var arrowY = oy + N * inCell / 2;
                            ctx.beginPath(); ctx.moveTo(ox + N * inCell + 8, arrowY); ctx.lineTo(kOx - 8, arrowY); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(kOx + 3 * kCell + 8, arrowY); ctx.lineTo(outOx - 8, arrowY); ctx.stroke();
                            ctx.setLineDash([]);

                            // Draw arrow heads
                            ctx.fillStyle = filt.color + '66';
                            ctx.beginPath();
                            ctx.moveTo(kOx - 8, arrowY);
                            ctx.lineTo(kOx - 16, arrowY - 5);
                            ctx.lineTo(kOx - 16, arrowY + 5);
                            ctx.closePath(); ctx.fill();
                            ctx.beginPath();
                            ctx.moveTo(outOx - 8, arrowY);
                            ctx.lineTo(outOx - 16, arrowY - 5);
                            ctx.lineTo(outOx - 16, arrowY + 5);
                            ctx.closePath(); ctx.fill();
                        }

                        draw();
                        return viz;
                    }
                }
            ],

            exercises: [
                {
                    question: 'Show that the identity kernel \\(\\begin{pmatrix}0 & 0 & 0\\\\0 & 1 & 0\\\\0 & 0 & 0\\end{pmatrix}\\) leaves the input unchanged (up to valid-padding size reduction). What kernel would shift the image one pixel to the right?',
                    hint: 'For the identity, each output position copies the center element of its receptive field. For a shift, think about which element in the receptive field should be copied.',
                    solution: 'The identity kernel computes \\(Y[i,j] = \\sum_{m,n} K[m,n] \\cdot X[i+m, j+n] = 1 \\cdot X[i+1, j+1] = X[i+1, j+1]\\), which is the original image (shifted by the 1-pixel border lost to valid padding). To shift right by one pixel, we need \\(Y[i,j] = X[i+1, j]\\), which corresponds to the kernel \\(\\begin{pmatrix}0 & 0 & 0\\\\1 & 0 & 0\\\\0 & 0 & 0\\end{pmatrix}\\).'
                },
                {
                    question: 'The sharpen kernel can be written as \\(K_{\\text{sharp}} = \\alpha K_{\\text{identity}} + \\beta K_{\\text{laplacian}}\\) where \\(K_{\\text{laplacian}} = \\begin{pmatrix}0 & -1 & 0\\\\-1 & 4 & -1\\\\0 & -1 & 0\\end{pmatrix}\\). Find \\(\\alpha\\) and \\(\\beta\\). What happens if you increase \\(\\beta\\)?',
                    hint: 'The identity kernel is \\(\\begin{pmatrix}0 & 0 & 0\\\\0 & 1 & 0\\\\0 & 0 & 0\\end{pmatrix}\\). Set up the equation \\(\\alpha I + \\beta L = K_{\\text{sharp}}\\) and match entries.',
                    solution: 'Matching the center: \\(\\alpha + 4\\beta = 5\\). Matching an off-center entry: \\(-\\beta = -1\\), so \\(\\beta = 1\\) and \\(\\alpha = 1\\). Thus \\(K_{\\text{sharp}} = I + L\\). Increasing \\(\\beta\\) amplifies the Laplacian component, producing more aggressive sharpening. At high \\(\\beta\\), the output becomes dominated by edges (essentially an edge detector), and flat regions get suppressed.'
                },
                {
                    question: 'The Sobel operators are <em>separable</em>: \\(K_v = \\begin{pmatrix}1\\\\2\\\\1\\end{pmatrix}\\begin{pmatrix}-1 & 0 & 1\\end{pmatrix}\\). Why is separability computationally advantageous? How many multiplications does the separable form use per output pixel vs. the full \\(3 \\times 3\\) kernel?',
                    hint: 'A separable filter of size \\(k \\times k\\) can be applied as two 1D convolutions: one \\(k \\times 1\\) and one \\(1 \\times k\\).',
                    solution: 'A full \\(3 \\times 3\\) kernel requires \\(3 \\times 3 = 9\\) multiplications per output pixel. The separable form applies a \\(3 \\times 1\\) filter (3 multiplications) followed by a \\(1 \\times 3\\) filter (3 multiplications), totaling 6 multiplications per pixel. In general, separability reduces a \\(k \\times k\\) convolution from \\(k^2\\) to \\(2k\\) operations per pixel. For \\(k = 7\\), that is 49 vs. 14, a 3.5x speedup. This principle is the foundation of depthwise separable convolutions used in MobileNet.'
                }
            ]
        },

        // ===================== Section 5: Advanced Convolutions =====================
        {
            id: 'ch09-sec05',
            title: 'Advanced Convolutions',
            content: `<h2>Advanced Convolutions</h2>

                <p>Beyond the standard convolution, several variants address specific design goals: reducing computation, changing spatial dimensions, or mixing channels efficiently.</p>

                <h3>1&times;1 Convolution (Pointwise Convolution)</h3>

                <div class="env-block definition">
                    <div class="env-title">Definition (1&times;1 Convolution)</div>
                    <div class="env-body"><p>A <strong>1&times;1 convolution</strong> uses kernels of spatial size \\(1 \\times 1\\). Each output at position \\((i,j)\\) is a linear combination of all \\(C_{\\text{in}}\\) input channels at that same position:
                    \\[Y[f, i, j] = \\sum_{c=0}^{C_{\\text{in}}-1} K[f, c] \\cdot X[c, i, j] + b[f]\\]
                    where \\(f\\) indexes the output filter. This is equivalent to applying a shared fully connected layer independently at each spatial location. A \\(1 \\times 1\\) convolution with \\(C_{\\text{out}}\\) filters transforms \\(\\mathbb{R}^{C_{\\text{in}} \\times H \\times W} \\to \\mathbb{R}^{C_{\\text{out}} \\times H \\times W}\\) using \\(C_{\\text{in}} \\times C_{\\text{out}} + C_{\\text{out}}\\) parameters.</p></div>
                </div>

                <div class="env-block remark">
                    <div class="env-title">Remark (Uses of 1&times;1 Convolution)</div>
                    <div class="env-body"><p><strong>Channel reduction (bottleneck):</strong> In GoogLeNet/Inception, 1&times;1 convolutions reduce the channel dimension before expensive \\(3 \\times 3\\) or \\(5 \\times 5\\) convolutions, cutting computation by 2-10x. <strong>Channel expansion:</strong> In ResNet bottleneck blocks, 1&times;1 convs expand channels back after a narrow \\(3 \\times 3\\) layer. <strong>Cross-channel mixing:</strong> Even when the channel count stays the same, 1&times;1 convolutions learn nonlinear combinations of features (when followed by an activation), effectively acting as a learned projection in channel space.</p></div>
                </div>

                <h3>Depthwise Separable Convolution</h3>

                <div class="env-block definition">
                    <div class="env-title">Definition (Depthwise Separable Convolution)</div>
                    <div class="env-body"><p>A <strong>depthwise separable convolution</strong> factorizes a standard convolution into two steps:</p>
                    <ol>
                        <li><strong>Depthwise convolution:</strong> Apply a separate \\(k \\times k\\) filter to each input channel independently. This produces \\(C_{\\text{in}}\\) feature maps, one per channel. Parameters: \\(C_{\\text{in}} \\times k^2\\).</li>
                        <li><strong>Pointwise convolution:</strong> Apply a \\(1 \\times 1\\) convolution with \\(C_{\\text{out}}\\) filters to mix channels. Parameters: \\(C_{\\text{in}} \\times C_{\\text{out}}\\).</li>
                    </ol>
                    <p>Total parameters: \\(C_{\\text{in}} \\times k^2 + C_{\\text{in}} \\times C_{\\text{out}}\\).</p></div>
                </div>

                <div class="env-block theorem">
                    <div class="env-title">Proposition (Parameter Reduction Ratio)</div>
                    <div class="env-body"><p>The ratio of depthwise separable to standard convolution parameters is
                    \\[\\frac{C_{\\text{in}} k^2 + C_{\\text{in}} C_{\\text{out}}}{C_{\\text{in}} C_{\\text{out}} k^2} = \\frac{1}{C_{\\text{out}}} + \\frac{1}{k^2}\\]
                    For typical values (\\(C_{\\text{out}} = 256\\), \\(k = 3\\)), this is approximately \\(1/256 + 1/9 \\approx 0.115\\), roughly an <strong>8-9x reduction</strong> in parameters and computation.</p></div>
                </div>

                <div class="env-block proof">
                    <div class="env-title">Proof</div>
                    <div class="env-body"><p>Standard convolution: \\(C_{\\text{in}} \\cdot C_{\\text{out}} \\cdot k^2\\) parameters (ignoring bias). Depthwise separable: \\(C_{\\text{in}} \\cdot k^2 + C_{\\text{in}} \\cdot C_{\\text{out}}\\). The ratio is
                    \\[\\frac{C_{\\text{in}} k^2 + C_{\\text{in}} C_{\\text{out}}}{C_{\\text{in}} C_{\\text{out}} k^2} = \\frac{k^2 + C_{\\text{out}}}{C_{\\text{out}} k^2} = \\frac{1}{C_{\\text{out}}} + \\frac{1}{k^2}\\]</p>
                    <div class="qed">&#8718;</div></div>
                </div>

                <div class="env-block example">
                    <div class="env-title">Example (MobileNet vs. Standard Conv)</div>
                    <div class="env-body"><p>Consider a layer with \\(C_{\\text{in}} = 128\\), \\(C_{\\text{out}} = 256\\), \\(k = 3\\).</p>
                    <p><strong>Standard conv:</strong> \\(128 \\times 256 \\times 9 = 294{,}912\\) parameters.</p>
                    <p><strong>Depthwise separable:</strong> \\(128 \\times 9 + 128 \\times 256 = 1{,}152 + 32{,}768 = 33{,}920\\) parameters.</p>
                    <p>Reduction factor: \\(294{,}912 / 33{,}920 \\approx 8.7\\times\\). MobileNet (Howard et al., 2017) uses this factorization throughout, enabling CNNs to run on mobile devices.</p></div>
                </div>

                <h3>Transposed Convolution (Deconvolution)</h3>

                <div class="env-block definition">
                    <div class="env-title">Definition (Transposed Convolution)</div>
                    <div class="env-body"><p>A <strong>transposed convolution</strong> (sometimes misleadingly called "deconvolution") is the gradient operation of a strided convolution. It <em>upsamples</em> the spatial dimensions. Given an input of size \\(H_{\\text{in}}\\), the transposed convolution with kernel \\(k\\), stride \\(s\\), and padding \\(p\\) produces an output of size
                    \\[H_{\\text{out}} = (H_{\\text{in}} - 1) \\cdot s - 2p + k\\]
                    Intuitively, a transposed convolution inserts \\(s - 1\\) zeros between each input element and then applies a standard convolution. It is used in architectures that require spatial upsampling, such as decoder networks in autoencoders and U-Net.</p></div>
                </div>

                <div class="env-block warning">
                    <div class="env-title">Warning (Checkerboard Artifacts)</div>
                    <div class="env-body"><p>Transposed convolutions with stride &gt; 1 are prone to <strong>checkerboard artifacts</strong> when the kernel size is not divisible by the stride. Overlapping and non-uniform coverage of the output positions creates a periodic pattern. The remedy: use kernel sizes divisible by the stride (e.g., \\(k=4, s=2\\)), or replace transposed convolutions with bilinear upsampling followed by a regular convolution (Odena et al., 2016).</p></div>
                </div>

                <div class="viz-placeholder" data-viz="viz-param-compare"></div>

                <div class="env-block intuition">
                    <div class="env-title">Design Philosophy</div>
                    <div class="env-body"><p>The evolution from standard to depthwise separable convolution reflects a general principle in neural architecture design: <strong>decouple spatial and channel processing</strong>. Standard convolution jointly mixes spatial neighborhoods and channels. Depthwise separable convolution decouples them into independent steps, which turns out to lose little representational power while saving enormous computation. This principle extends to architectures like Transformers, where spatial (attention) and channel (MLP) processing are also separated.</p></div>
                </div>`,

            visualizations: [
                {
                    id: 'viz-param-compare',
                    title: 'Parameter Count: Standard vs. Depthwise Separable Convolution',
                    description: 'Adjust C_in, C_out, and kernel size to compare parameter counts. The bar chart shows the dramatic reduction from depthwise separable convolution.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 680, height: 400, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;

                        var cIn = 64;
                        var cOut = 128;
                        var kSize = 3;

                        VizEngine.createSlider(controls, 'C_in', 16, 512, cIn, 16, function(v) { cIn = Math.round(v); draw(); });
                        VizEngine.createSlider(controls, 'C_out', 16, 512, cOut, 16, function(v) { cOut = Math.round(v); draw(); });
                        VizEngine.createSlider(controls, 'Kernel', 1, 7, kSize, 2, function(v) { kSize = Math.round(v); if (kSize % 2 === 0) kSize++; draw(); });

                        function draw() {
                            viz.clear();

                            var stdParams = cIn * cOut * kSize * kSize;
                            var dwParams = cIn * kSize * kSize;
                            var pwParams = cIn * cOut;
                            var sepParams = dwParams + pwParams;
                            var ratio = stdParams / Math.max(sepParams, 1);

                            // Bar chart area
                            var barLeft = 100;
                            var barTop = 60;
                            var barHeight = 50;
                            var barGap = 30;
                            var maxWidth = 480;
                            var maxVal = stdParams;

                            function drawBar(y, val, maxV, color, label, subLabel) {
                                var w = (val / maxV) * maxWidth;
                                w = Math.max(w, 2);
                                ctx.fillStyle = color;
                                ctx.fillRect(barLeft, y, w, barHeight);
                                ctx.strokeStyle = color;
                                ctx.lineWidth = 1;
                                ctx.strokeRect(barLeft, y, w, barHeight);

                                ctx.fillStyle = '#f0f6fc';
                                ctx.font = 'bold 13px -apple-system,sans-serif';
                                ctx.textAlign = 'right';
                                ctx.textBaseline = 'middle';
                                ctx.fillText(label, barLeft - 10, y + barHeight / 2);

                                ctx.fillStyle = '#c9d1d9';
                                ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                var valText = val >= 1000000 ? (val / 1000000).toFixed(2) + 'M' : (val >= 1000 ? (val / 1000).toFixed(1) + 'K' : val);
                                ctx.fillText(valText, barLeft + w + 8, y + barHeight / 2);

                                if (subLabel) {
                                    ctx.fillStyle = '#6e7681';
                                    ctx.font = '10px -apple-system,sans-serif';
                                    ctx.fillText(subLabel, barLeft + w + 8, y + barHeight / 2 + 15);
                                }
                            }

                            // Title
                            ctx.fillStyle = '#f0f6fc';
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Parameter Count Comparison', viz.width / 2, 30);

                            ctx.fillStyle = '#8b949e';
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.fillText('C_in=' + cIn + '  C_out=' + cOut + '  k=' + kSize + '\u00d7' + kSize, viz.width / 2, 48);

                            drawBar(barTop, stdParams, maxVal, '#f0883e', 'Standard',
                                cIn + '\u00d7' + cOut + '\u00d7' + kSize + '\u00b2 = ' + stdParams.toLocaleString());

                            // Depthwise separable as stacked bar
                            var sepY = barTop + barHeight + barGap;
                            var dwW = (dwParams / maxVal) * maxWidth;
                            var pwW = (pwParams / maxVal) * maxWidth;
                            dwW = Math.max(dwW, 2);
                            pwW = Math.max(pwW, 2);

                            ctx.fillStyle = '#3fb9a0';
                            ctx.fillRect(barLeft, sepY, dwW, barHeight);
                            ctx.fillStyle = '#58a6ff';
                            ctx.fillRect(barLeft + dwW, sepY, pwW, barHeight);
                            ctx.strokeStyle = '#3fb9a0';
                            ctx.lineWidth = 1;
                            ctx.strokeRect(barLeft, sepY, dwW + pwW, barHeight);

                            ctx.fillStyle = '#f0f6fc';
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Separable', barLeft - 10, sepY + barHeight / 2);

                            var sepText = sepParams >= 1000000 ? (sepParams / 1000000).toFixed(2) + 'M' : (sepParams >= 1000 ? (sepParams / 1000).toFixed(1) + 'K' : sepParams);
                            ctx.fillStyle = '#c9d1d9';
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText(sepText, barLeft + dwW + pwW + 8, sepY + barHeight / 2);

                            ctx.fillStyle = '#6e7681';
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillText(cIn + '\u00d7' + kSize + '\u00b2 + ' + cIn + '\u00d7' + cOut + ' = ' + sepParams.toLocaleString(), barLeft + dwW + pwW + 8, sepY + barHeight / 2 + 15);

                            // Legend
                            var legY = sepY + barHeight + 30;
                            ctx.fillStyle = '#3fb9a0';
                            ctx.fillRect(barLeft, legY, 14, 14);
                            ctx.fillStyle = '#c9d1d9';
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('Depthwise (' + dwParams.toLocaleString() + ')', barLeft + 20, legY + 10);

                            ctx.fillStyle = '#58a6ff';
                            ctx.fillRect(barLeft + 180, legY, 14, 14);
                            ctx.fillStyle = '#c9d1d9';
                            ctx.fillText('Pointwise (' + pwParams.toLocaleString() + ')', barLeft + 200, legY + 10);

                            // Ratio box
                            var boxY = legY + 36;
                            ctx.fillStyle = '#1a1a40';
                            ctx.fillRect(barLeft, boxY, 340, 50);
                            ctx.strokeStyle = '#30363d';
                            ctx.lineWidth = 1;
                            ctx.strokeRect(barLeft, boxY, 340, 50);

                            ctx.fillStyle = '#f0883e';
                            ctx.font = 'bold 16px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Reduction: ' + ratio.toFixed(1) + '\u00d7 fewer parameters', barLeft + 170, boxY + 20);

                            ctx.fillStyle = '#8b949e';
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.fillText('Ratio = 1/C_out + 1/k\u00b2 = 1/' + cOut + ' + 1/' + (kSize * kSize) + ' \u2248 ' + (1 / cOut + 1 / (kSize * kSize)).toFixed(4), barLeft + 170, boxY + 40);

                            // Computation diagram at bottom
                            var diagY = boxY + 70;
                            ctx.fillStyle = '#f0f6fc';
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Depthwise Separable = Depthwise + Pointwise', viz.width / 2, diagY);

                            // Depthwise box
                            var dwBoxX = 60, dwBoxW = 220, dwBoxH = 45;
                            ctx.fillStyle = 'rgba(63,185,160,0.1)';
                            ctx.fillRect(dwBoxX, diagY + 12, dwBoxW, dwBoxH);
                            ctx.strokeStyle = '#3fb9a0';
                            ctx.lineWidth = 1.5;
                            ctx.strokeRect(dwBoxX, diagY + 12, dwBoxW, dwBoxH);
                            ctx.fillStyle = '#3fb9a0';
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillText('Depthwise: ' + kSize + '\u00d7' + kSize + ' per channel', dwBoxX + dwBoxW / 2, diagY + 27);
                            ctx.fillStyle = '#8b949e';
                            ctx.fillText(cIn + ' channels \u00d7 ' + kSize + '\u00b2 = ' + dwParams + ' params', dwBoxX + dwBoxW / 2, diagY + 44);

                            // Arrow
                            ctx.fillStyle = '#6e7681';
                            ctx.font = '18px -apple-system,sans-serif';
                            ctx.fillText('\u2192', dwBoxX + dwBoxW + 15, diagY + 36);

                            // Pointwise box
                            var pwBoxX = dwBoxX + dwBoxW + 40;
                            ctx.fillStyle = 'rgba(88,166,255,0.1)';
                            ctx.fillRect(pwBoxX, diagY + 12, dwBoxW, dwBoxH);
                            ctx.strokeStyle = '#58a6ff';
                            ctx.lineWidth = 1.5;
                            ctx.strokeRect(pwBoxX, diagY + 12, dwBoxW, dwBoxH);
                            ctx.fillStyle = '#58a6ff';
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillText('Pointwise: 1\u00d71, mix channels', pwBoxX + dwBoxW / 2, diagY + 27);
                            ctx.fillStyle = '#8b949e';
                            ctx.fillText(cIn + ' \u00d7 ' + cOut + ' = ' + pwParams.toLocaleString() + ' params', pwBoxX + dwBoxW / 2, diagY + 44);
                        }

                        draw();
                        return viz;
                    }
                }
            ],

            exercises: [
                {
                    question: 'A GoogLeNet Inception module applies \\(1 \\times 1\\) convolution to reduce \\(C_{\\text{in}} = 256\\) channels to 64 before a \\(5 \\times 5\\) convolution with \\(C_{\\text{out}} = 128\\). Compare the total parameter count (1&times;1 reduction + 5&times;5) to a direct \\(5 \\times 5\\) convolution from 256 to 128 channels.',
                    hint: 'The 1&times;1 layer maps 256 to 64 channels. The 5&times;5 then maps 64 to 128.',
                    solution: 'Direct \\(5 \\times 5\\): \\(256 \\times 128 \\times 25 = 819{,}200\\) parameters. With bottleneck: \\(256 \\times 64 \\times 1 + 64 \\times 128 \\times 25 = 16{,}384 + 204{,}800 = 221{,}184\\) parameters. The bottleneck reduces parameters by a factor of \\(819{,}200 / 221{,}184 \\approx 3.7\\times\\).'
                },
                {
                    question: 'Verify that a transposed convolution with \\(k = 4\\), \\(s = 2\\), \\(p = 1\\) upsamples a \\(7 \\times 7\\) feature map to \\(14 \\times 14\\). What is the general condition on \\(k\\), \\(s\\), and \\(p\\) for exact 2x upsampling (\\(H_{\\text{out}} = 2 H_{\\text{in}}\\))?',
                    hint: 'Use the transposed conv formula: \\(H_{\\text{out}} = (H_{\\text{in}} - 1) \\cdot s - 2p + k\\).',
                    solution: '\\(H_{\\text{out}} = (7 - 1) \\cdot 2 - 2 \\cdot 1 + 4 = 12 - 2 + 4 = 14\\). For exact 2x upsampling: \\(2H_{\\text{in}} = (H_{\\text{in}} - 1) \\cdot 2 - 2p + k = 2H_{\\text{in}} - 2 - 2p + k\\), so \\(k = 2 + 2p\\). With \\(s = 2\\): \\(p = 0 \\Rightarrow k = 2\\); \\(p = 1 \\Rightarrow k = 4\\); \\(p = 2 \\Rightarrow k = 6\\). The common choice \\(k = 2s\\) with \\(p = s/2\\) achieves exact 2x upsampling.'
                },
                {
                    question: 'In a depthwise separable convolution with \\(C_{\\text{in}} = C_{\\text{out}} = C\\) and kernel size \\(k\\), what value of \\(C\\) makes the depthwise component (\\(Ck^2\\) parameters) equal to the pointwise component (\\(C^2\\) parameters)? Interpret this threshold.',
                    hint: 'Set \\(Ck^2 = C^2\\) and solve for \\(C\\).',
                    solution: '\\(Ck^2 = C^2\\) gives \\(C = k^2\\). For \\(k = 3\\): \\(C = 9\\). When \\(C &lt; k^2\\), the depthwise part dominates (spatial processing is the bottleneck). When \\(C &gt; k^2\\), the pointwise part dominates (channel mixing is the bottleneck). Since modern networks use \\(C \\gg k^2\\) (e.g., \\(C = 256\\), \\(k = 3\\)), the pointwise convolution accounts for the vast majority of computation, which is why efficient architectures focus on reducing the channel mixing cost (e.g., through group convolutions or shuffle operations in ShuffleNet).'
                }
            ]
        }
    ]
});
