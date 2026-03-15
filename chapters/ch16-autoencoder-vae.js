window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch16',
    number: 16,
    title: 'Autoencoders & VAE',
    subtitle: 'From deterministic compression to probabilistic generative models via the reparameterization trick and ELBO',
    sections: [
        // ===================== Section 1: Autoencoders =====================
        {
            id: 'ch16-sec01',
            title: 'Autoencoders',
            content: `<h2>Autoencoders</h2>

                <div class="env-block intuition">
                    <div class="env-title">Why Autoencoders?</div>
                    <div class="env-body"><p>So far in this course, we have trained neural networks with <em>labeled</em> data: given input \\(\\mathbf{x}\\), predict target \\(\\mathbf{y}\\). But what if we want the network to discover structure in <em>unlabeled</em> data? An autoencoder does exactly this: it learns to compress the input into a compact representation and then reconstruct the input from that representation. Any information that survives this bottleneck must be, in some sense, the most important.</p></div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">Definition (Autoencoder)</div>
                    <div class="env-body"><p>An <strong>autoencoder</strong> is a neural network composed of two subnetworks:</p>
                    <ol>
                        <li>An <strong>encoder</strong> \\(f_\\phi: \\mathbb{R}^D \\to \\mathbb{R}^d\\), where \\(d \\ll D\\), that maps the input \\(\\mathbf{x}\\) to a <strong>latent representation</strong> (or <strong>code</strong>) \\(\\mathbf{z} = f_\\phi(\\mathbf{x})\\).</li>
                        <li>A <strong>decoder</strong> \\(g_\\theta: \\mathbb{R}^d \\to \\mathbb{R}^D\\) that maps the code back to an approximation of the input \\(\\hat{\\mathbf{x}} = g_\\theta(\\mathbf{z})\\).</li>
                    </ol>
                    <p>The integer \\(d\\) is the <strong>bottleneck dimension</strong> (or latent dimension). The parameters \\(\\phi\\) and \\(\\theta\\) are trained jointly.</p></div>
                </div>

                <p>The full autoencoder is the composition \\(\\hat{\\mathbf{x}} = g_\\theta(f_\\phi(\\mathbf{x}))\\). Training minimizes the discrepancy between input and reconstruction.</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (Reconstruction Loss)</div>
                    <div class="env-body"><p>The <strong>reconstruction loss</strong> measures how faithfully the decoder recovers the original input from the latent code. For continuous data, the standard choice is mean squared error (MSE):
                    \\[\\mathcal{L}_{\\text{recon}}(\\phi, \\theta) = \\frac{1}{N} \\sum_{i=1}^{N} \\|\\mathbf{x}^{(i)} - g_\\theta(f_\\phi(\\mathbf{x}^{(i)}))\\|^2\\]
                    For binary data (e.g., pixel values in \\([0,1]\\)), <strong>binary cross-entropy</strong> is often preferred:
                    \\[\\mathcal{L}_{\\text{BCE}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\sum_{j=1}^{D} \\bigl[ x_j^{(i)} \\log \\hat{x}_j^{(i)} + (1 - x_j^{(i)}) \\log(1 - \\hat{x}_j^{(i)}) \\bigr]\\]</p></div>
                </div>

                <h3>Architecture and the Information Bottleneck</h3>

                <p>The critical design choice is the bottleneck dimension \\(d\\). If \\(d \\geq D\\), the network can learn the identity function, which compresses nothing and learns no useful representation. The constraint \\(d \\ll D\\) forces the encoder to discard irrelevant variation and retain only the factors of variation that are most useful for reconstruction.</p>

                <div class="env-block example">
                    <div class="env-title">Example (Linear Autoencoder = PCA)</div>
                    <div class="env-body"><p>If both the encoder and decoder are <em>linear</em> maps (i.e., matrices) and the loss is MSE, then the optimal solution recovers <strong>PCA</strong> (principal component analysis). Specifically, the encoder learns the projection onto the top-\\(d\\) principal components of the data covariance matrix, and the decoder learns the reverse projection. This can be proved by showing the optimal linear reconstruction minimizes the Frobenius norm \\(\\|\\mathbf{X} - \\mathbf{X} \\mathbf{W} \\mathbf{W}^\\top\\|_F^2\\) subject to \\(\\mathbf{W} \\in \\mathbb{R}^{D \\times d}\\), \\(\\mathbf{W}^\\top \\mathbf{W} = \\mathbf{I}_d\\), which is solved by the eigenvectors of \\(\\mathbf{X}^\\top \\mathbf{X}\\) with the \\(d\\) largest eigenvalues.</p>
                    <p>The power of deep autoencoders is that they use <em>nonlinear</em> encoder and decoder functions, enabling them to capture nonlinear manifold structure that PCA misses.</p></div>
                </div>

                <div class="env-block theorem">
                    <div class="env-title">Theorem (Optimal Linear Autoencoder)</div>
                    <div class="env-body"><p>Let \\(\\mathbf{X} \\in \\mathbb{R}^{N \\times D}\\) be a centered data matrix. Among all linear autoencoders with bottleneck dimension \\(d\\), the one minimizing MSE reconstruction loss has encoder weights \\(\\mathbf{W}_e = \\mathbf{U}_d^\\top\\) and decoder weights \\(\\mathbf{W}_d = \\mathbf{U}_d\\), where \\(\\mathbf{U}_d \\in \\mathbb{R}^{D \\times d}\\) contains the top-\\(d\\) eigenvectors of the covariance matrix \\(\\frac{1}{N}\\mathbf{X}^\\top \\mathbf{X}\\). The minimum reconstruction error equals \\(\\sum_{j=d+1}^{D} \\lambda_j\\), where \\(\\lambda_1 \\geq \\cdots \\geq \\lambda_D\\) are the eigenvalues.</p></div>
                </div>

                <h3>Deep Autoencoders in Practice</h3>

                <p>A typical deep autoencoder stacks several hidden layers in the encoder with decreasing widths (e.g., 784 &rarr; 256 &rarr; 64 &rarr; 16) and a symmetric decoder (16 &rarr; 64 &rarr; 256 &rarr; 784). ReLU or similar activations are used in intermediate layers, and the final decoder layer uses sigmoid (for \\([0,1]\\) outputs) or linear activation (for unbounded outputs).</p>

                <div class="env-block remark">
                    <div class="env-title">Remark (Denoising Autoencoders)</div>
                    <div class="env-body"><p>A <strong>denoising autoencoder</strong> (Vincent et al., 2008) corrupts the input \\(\\tilde{\\mathbf{x}} = \\text{corrupt}(\\mathbf{x})\\) (e.g., by adding Gaussian noise or masking random features) and trains the network to reconstruct the <em>clean</em> input from the corrupted version: \\(\\min \\|\\mathbf{x} - g_\\theta(f_\\phi(\\tilde{\\mathbf{x}}))\\|^2\\). This prevents the network from merely learning the identity and forces it to learn robust features. Denoising autoencoders were an important precursor to modern generative models and are closely related to the score-matching perspective behind diffusion models (Chapter 18).</p></div>
                </div>

                <div class="viz-placeholder" data-viz="viz-autoencoder-arch"></div>`,

            visualizations: [
                {
                    id: 'viz-autoencoder-arch',
                    title: 'Autoencoder Architecture: Data Flow',
                    description: 'Animated data flow through an autoencoder. The input is compressed through the encoder into a low-dimensional bottleneck, then reconstructed by the decoder. Adjust the bottleneck dimension to see how it affects reconstruction fidelity.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, {width: 700, height: 400, scale: 1, originX: 0, originY: 0});
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var bottleneck = 2;
                        var animT = 0;

                        VizEngine.createSlider(controls, 'Bottleneck dim', 1, 5, 2, 1, function(v) { bottleneck = Math.round(v); });

                        var layerWidths = {
                            1: [8, 6, 4, 1, 4, 6, 8],
                            2: [8, 6, 4, 2, 4, 6, 8],
                            3: [8, 6, 4, 3, 4, 6, 8],
                            4: [8, 6, 5, 4, 5, 6, 8],
                            5: [8, 7, 6, 5, 6, 7, 8]
                        };

                        function getLayerSizes() {
                            return layerWidths[bottleneck] || layerWidths[2];
                        }

                        function drawNode(x, y, r, color, glow) {
                            if (glow) {
                                ctx.fillStyle = color + '33';
                                ctx.beginPath(); ctx.arc(x, y, r + 6, 0, Math.PI * 2); ctx.fill();
                            }
                            ctx.fillStyle = color;
                            ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
                            ctx.strokeStyle = color;
                            ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();
                        }

                        function draw(t) {
                            animT = (t / 2000) % 1;
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var sizes = getLayerSizes();
                            var numLayers = sizes.length;
                            var xMargin = 70;
                            var xStep = (W - 2 * xMargin) / (numLayers - 1);
                            var nodeR = 9;
                            var ySpacing = 32;
                            var midLayer = Math.floor(numLayers / 2);

                            // Draw connections first (behind nodes)
                            for (var l = 0; l < numLayers - 1; l++) {
                                var x1 = xMargin + l * xStep;
                                var x2 = xMargin + (l + 1) * xStep;
                                for (var i = 0; i < sizes[l]; i++) {
                                    var y1 = H / 2 + (i - (sizes[l] - 1) / 2) * ySpacing;
                                    for (var j = 0; j < sizes[l + 1]; j++) {
                                        var y2 = H / 2 + (j - (sizes[l + 1] - 1) / 2) * ySpacing;
                                        var alpha = 0.08;
                                        // Highlight connections near the animated "data packet"
                                        var packetLayer = animT * (numLayers - 1);
                                        if (Math.abs(l - packetLayer) < 0.8) {
                                            alpha = 0.25;
                                        }
                                        ctx.strokeStyle = 'rgba(88,166,255,' + alpha + ')';
                                        ctx.lineWidth = 0.8;
                                        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
                                    }
                                }
                            }

                            // Draw nodes
                            for (var l = 0; l < numLayers; l++) {
                                var x = xMargin + l * xStep;
                                var isBottleneck = (l === midLayer);
                                var isEncoder = (l < midLayer);
                                var isDecoder = (l > midLayer);

                                for (var i = 0; i < sizes[l]; i++) {
                                    var y = H / 2 + (i - (sizes[l] - 1) / 2) * ySpacing;
                                    var color;
                                    if (isBottleneck) color = viz.colors.orange;
                                    else if (isEncoder) color = viz.colors.blue;
                                    else color = viz.colors.teal;

                                    var packetLayer = animT * (numLayers - 1);
                                    var glow = Math.abs(l - packetLayer) < 0.6;
                                    drawNode(x, y, nodeR, color, glow);
                                }
                            }

                            // Draw data packet animation
                            var packetLayer = animT * (numLayers - 1);
                            var layerIdx = Math.floor(packetLayer);
                            var frac = packetLayer - layerIdx;
                            if (layerIdx < numLayers - 1) {
                                var x1 = xMargin + layerIdx * xStep;
                                var x2 = xMargin + (layerIdx + 1) * xStep;
                                var px = x1 + (x2 - x1) * frac;
                                var py = H / 2;
                                ctx.fillStyle = viz.colors.yellow + 'cc';
                                ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.fill();
                                ctx.fillStyle = viz.colors.yellow;
                                ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
                            }

                            // Labels
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';

                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillText('Encoder f\u03D5', xMargin + 1 * xStep, H - 30);

                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillText('Bottleneck (d=' + bottleneck + ')', xMargin + midLayer * xStep, H - 30);

                            ctx.fillStyle = viz.colors.teal;
                            ctx.fillText('Decoder g\u03B8', xMargin + (numLayers - 2) * xStep, H - 30);

                            // Input / output labels
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.text;
                            ctx.textAlign = 'center';
                            ctx.fillText('Input x', xMargin, 18);
                            ctx.fillText('(D=' + sizes[0] + ')', xMargin, 33);
                            ctx.fillText('Output x\u0302', xMargin + (numLayers - 1) * xStep, 18);
                            ctx.fillText('(D=' + sizes[numLayers - 1] + ')', xMargin + (numLayers - 1) * xStep, 33);

                            // Latent code label
                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillText('z', xMargin + midLayer * xStep, 18);
                            ctx.fillText('(d=' + bottleneck + ')', xMargin + midLayer * xStep, 33);

                            // Brackets for encoder / decoder
                            ctx.strokeStyle = viz.colors.blue + '66';
                            ctx.lineWidth = 2;
                            ctx.setLineDash([4, 3]);
                            var encLeft = xMargin - 20;
                            var encRight = xMargin + midLayer * xStep - 20;
                            ctx.beginPath();
                            ctx.moveTo(encLeft, 50); ctx.lineTo(encRight, 50);
                            ctx.stroke();

                            ctx.strokeStyle = viz.colors.teal + '66';
                            var decLeft = xMargin + midLayer * xStep + 20;
                            var decRight = xMargin + (numLayers - 1) * xStep + 20;
                            ctx.beginPath();
                            ctx.moveTo(decLeft, 50); ctx.lineTo(decRight, 50);
                            ctx.stroke();
                            ctx.setLineDash([]);
                        }

                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Suppose you train a linear autoencoder (no activation functions) with encoder \\(\\mathbf{W}_e \\in \\mathbb{R}^{d \\times D}\\) and decoder \\(\\mathbf{W}_d \\in \\mathbb{R}^{D \\times d}\\) to minimize MSE on centered data. Show that the optimal \\(\\mathbf{W}_d \\mathbf{W}_e\\) is the projection onto the subspace spanned by the top-\\(d\\) eigenvectors of the data covariance.',
                    hint: 'Write the loss as \\(\\|\\mathbf{X} - \\mathbf{X} \\mathbf{W}_e^\\top \\mathbf{W}_d^\\top\\|_F^2\\). Use the Eckart-Young theorem (best rank-\\(d\\) approximation in Frobenius norm is given by the truncated SVD).',
                    solution: 'The reconstruction is \\(\\hat{\\mathbf{X}} = \\mathbf{X} \\mathbf{W}_e^\\top \\mathbf{W}_d^\\top\\). The matrix \\(\\mathbf{P} = \\mathbf{W}_e^\\top \\mathbf{W}_d^\\top\\) has rank at most \\(d\\). By the Eckart-Young theorem, \\(\\|\\mathbf{X} - \\mathbf{X}\\mathbf{P}\\|_F^2\\) is minimized when \\(\\mathbf{P}\\) is the projection onto the top-\\(d\\) right singular vectors of \\(\\mathbf{X}\\), which are the top-\\(d\\) eigenvectors of \\(\\mathbf{X}^\\top \\mathbf{X}\\) (proportional to the covariance for centered data). Hence optimal linear AE = PCA.'
                },
                {
                    question: 'A denoising autoencoder adds Gaussian noise \\(\\tilde{\\mathbf{x}} = \\mathbf{x} + \\boldsymbol{\\epsilon}\\) with \\(\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(0, \\sigma^2 \\mathbf{I})\\) and minimizes \\(\\mathbb{E}[\\|\\mathbf{x} - g_\\theta(f_\\phi(\\tilde{\\mathbf{x}}))\\|^2]\\). Explain intuitively why this prevents learning the identity function, even when \\(d = D\\).',
                    hint: 'Think about what happens if the network memorizes or copies its input directly.',
                    solution: 'If the network learned the identity, it would output \\(\\tilde{\\mathbf{x}} = \\mathbf{x} + \\boldsymbol{\\epsilon}\\), incurring expected loss \\(\\mathbb{E}[\\|\\boldsymbol{\\epsilon}\\|^2] = D\\sigma^2\\). To do better, the network must learn to "denoise," that is, infer the clean signal from the corrupted observation. This requires learning the structure of the data distribution (where the data concentrates on the manifold), which is exactly the useful representation we want.'
                },
                {
                    question: 'Consider two autoencoders on MNIST (D=784): one with bottleneck \\(d=2\\) and one with \\(d=32\\). Which will have lower reconstruction loss? Which will produce more interpretable latent representations? Justify your answers.',
                    hint: 'Think about the capacity vs. interpretability trade-off.',
                    solution: 'The \\(d=32\\) autoencoder will have <em>lower reconstruction loss</em> because it has more capacity to preserve information through the bottleneck. However, the \\(d=2\\) autoencoder often produces <em>more interpretable</em> latent representations: with only 2 dimensions, every latent variable must encode a meaningful factor of variation (such as digit identity or stroke angle), and the latent space can be visualized directly as a 2D scatter plot. With \\(d=32\\), the representation is distributed across many dimensions and harder to interpret directly.'
                }
            ]
        },

        // ===================== Section 2: Latent Space =====================
        {
            id: 'ch16-sec02',
            title: 'Latent Space',
            content: `<h2>Latent Space</h2>

                <div class="env-block intuition">
                    <div class="env-title">From Compression to Representation</div>
                    <div class="env-body"><p>The latent code \\(\\mathbf{z} = f_\\phi(\\mathbf{x})\\) lives in a \\(d\\)-dimensional space called the <strong>latent space</strong>. This space is not just a compression artifact; it is a <em>learned representation</em> of the data. If the autoencoder works well, nearby points in latent space correspond to semantically similar inputs, and the structure of the latent space reveals the underlying factors of variation in the data.</p></div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">Definition (Latent Space)</div>
                    <div class="env-body"><p>The <strong>latent space</strong> \\(\\mathcal{Z} \\subseteq \\mathbb{R}^d\\) is the range of the encoder function \\(f_\\phi\\). Each point \\(\\mathbf{z} \\in \\mathcal{Z}\\) is a <strong>latent representation</strong> (or <strong>embedding</strong>) of some data point \\(\\mathbf{x}\\). The decoder \\(g_\\theta\\) maps latent points back to data space, defining a <strong>generative mapping</strong> \\(\\mathbf{z} \\mapsto g_\\theta(\\mathbf{z})\\).</p></div>
                </div>

                <h3>Dimensionality Reduction as Manifold Learning</h3>

                <p>High-dimensional data (images, text, audio) typically does not fill the full ambient space \\(\\mathbb{R}^D\\). Instead, it concentrates near a low-dimensional <strong>manifold</strong>. Consider images of handwritten digits: each image has 784 pixels, but the intrinsic degrees of freedom (digit identity, stroke width, slant, size) are far fewer. The autoencoder's task is to discover this manifold and parameterize it by the latent coordinates \\(\\mathbf{z}\\).</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (Data Manifold Hypothesis)</div>
                    <div class="env-body"><p>The <strong>manifold hypothesis</strong> states that real-world high-dimensional data concentrates near a low-dimensional manifold \\(\\mathcal{M} \\subset \\mathbb{R}^D\\) with intrinsic dimension \\(d \\ll D\\). An ideal autoencoder learns a parameterization of this manifold: the encoder maps data to manifold coordinates, and the decoder maps coordinates back to the ambient space.</p></div>
                </div>

                <h3>Structure of the Latent Space</h3>

                <p>When we train an autoencoder on data with known class structure (like digit classes 0-9), we can visualize the 2D latent space by coloring points according to their class. A well-trained autoencoder produces <strong>clusters</strong>: each digit class occupies a distinct region. However, a standard autoencoder makes no guarantee about the global structure of these clusters.</p>

                <div class="env-block remark">
                    <div class="env-title">Remark (Limitations of AE Latent Space)</div>
                    <div class="env-body"><p>A deterministic autoencoder has two significant limitations:</p>
                    <ol>
                        <li><strong>Gaps and discontinuities.</strong> The encoder maps only training data to specific points. Regions between clusters are "uncharted territory" where the decoder may produce meaningless outputs.</li>
                        <li><strong>No principled generation.</strong> Unlike a generative model, there is no natural probability distribution over \\(\\mathcal{Z}\\) to sample from. If we pick a random \\(\\mathbf{z}\\), the decoded output \\(g_\\theta(\\mathbf{z})\\) may be garbage.</li>
                    </ol>
                    <p>These limitations motivate the variational autoencoder (Section 3).</p></div>
                </div>

                <div class="env-block example">
                    <div class="env-title">Example (t-SNE vs. Autoencoder Embeddings)</div>
                    <div class="env-body"><p>Both t-SNE and autoencoders produce 2D representations, but they differ fundamentally. t-SNE optimizes a divergence between neighborhood distributions and produces embeddings that are good for <em>visualization</em> but have no decoder: you cannot map a new point from 2D back to the original space. An autoencoder, by contrast, learns both an encoder <em>and</em> a decoder, making the embedding <strong>invertible</strong> (approximately). This invertibility is what makes autoencoders useful as generative models once we learn to sample from the latent space.</p></div>
                </div>

                <h3>Interpolation in Latent Space</h3>

                <p>A key test of latent space quality is <strong>interpolation</strong>. Given two data points \\(\\mathbf{x}_1\\) and \\(\\mathbf{x}_2\\) with latent codes \\(\\mathbf{z}_1 = f_\\phi(\\mathbf{x}_1)\\) and \\(\\mathbf{z}_2 = f_\\phi(\\mathbf{x}_2)\\), we form intermediate codes</p>
                \\[\\mathbf{z}_\\alpha = (1 - \\alpha)\\mathbf{z}_1 + \\alpha\\,\\mathbf{z}_2, \\qquad \\alpha \\in [0, 1]\\]
                <p>and decode each \\(g_\\theta(\\mathbf{z}_\\alpha)\\). In a good latent space, the decoded outputs should transition smoothly (e.g., from digit "3" to digit "8" through plausible intermediate forms). In a poorly structured latent space, intermediate decodings may be blurry or unrealistic.</p>

                <div class="viz-placeholder" data-viz="viz-latent-space"></div>`,

            visualizations: [
                {
                    id: 'viz-latent-space',
                    title: '2D Latent Space Visualization',
                    description: 'Simulated latent space of an autoencoder trained on digit data. Each colored cluster represents a digit class. Click "Resample" to generate a new random arrangement. Notice how clusters are separated but have gaps between them.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, {width: 600, height: 500, scale: 45, originX: 300, originY: 250});
                        var ctx = viz.ctx;

                        var digitColors = [
                            '#f85149', '#f0883e', '#d29922', '#3fb950', '#3fb9a0',
                            '#58a6ff', '#bc8cff', '#f778ba', '#8b949e', '#f0f6fc'
                        ];
                        var digitNames = ['0','1','2','3','4','5','6','7','8','9'];

                        // Generate cluster centers and points
                        var clusters = [];
                        var allPoints = [];

                        function generateData() {
                            clusters = [];
                            allPoints = [];
                            // Place 10 cluster centers in a rough circle arrangement
                            for (var c = 0; c < 10; c++) {
                                var angle = c * Math.PI * 2 / 10 + (Math.random() - 0.5) * 0.3;
                                var radius = 2.5 + (Math.random() - 0.5) * 0.8;
                                var cx = radius * Math.cos(angle);
                                var cy = radius * Math.sin(angle);
                                clusters.push({cx: cx, cy: cy, color: digitColors[c], label: digitNames[c]});

                                // Generate points around each center
                                var n = 40 + Math.floor(Math.random() * 20);
                                for (var i = 0; i < n; i++) {
                                    // Box-Muller for Gaussian
                                    var u1 = Math.random(), u2 = Math.random();
                                    var g1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                                    var g2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
                                    var spread = 0.35 + Math.random() * 0.15;
                                    allPoints.push({
                                        x: cx + g1 * spread,
                                        y: cy + g2 * spread,
                                        cls: c
                                    });
                                }
                            }
                        }

                        generateData();

                        VizEngine.createButton(controls, 'Resample', function() { generateData(); draw(); });

                        function draw() {
                            viz.clear();
                            viz.drawGrid();
                            viz.drawAxes();

                            // Draw all points
                            for (var i = 0; i < allPoints.length; i++) {
                                var p = allPoints[i];
                                var sx = viz.originX + p.x * viz.scale;
                                var sy = viz.originY - p.y * viz.scale;
                                ctx.fillStyle = digitColors[p.cls] + '99';
                                ctx.beginPath(); ctx.arc(sx, sy, 3, 0, Math.PI * 2); ctx.fill();
                            }

                            // Draw cluster labels at centers
                            ctx.font = 'bold 16px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            for (var c = 0; c < clusters.length; c++) {
                                var cl = clusters[c];
                                var sx = viz.originX + cl.cx * viz.scale;
                                var sy = viz.originY - cl.cy * viz.scale;
                                // Background circle
                                ctx.fillStyle = cl.color + '44';
                                ctx.beginPath(); ctx.arc(sx, sy, 16, 0, Math.PI * 2); ctx.fill();
                                ctx.fillStyle = cl.color;
                                ctx.fillText(cl.label, sx, sy);
                            }

                            // Legend
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'middle';
                            for (var c = 0; c < 10; c++) {
                                var lx = viz.width - 70;
                                var ly = 20 + c * 16;
                                ctx.fillStyle = digitColors[c];
                                ctx.beginPath(); ctx.arc(lx, ly, 4, 0, Math.PI * 2); ctx.fill();
                                ctx.fillText('Digit ' + c, lx + 10, ly);
                            }

                            // Title annotation
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.text;
                            ctx.textAlign = 'left';
                            ctx.fillText('z\u2081', viz.width - 25, viz.originY + 15);
                            ctx.fillText('z\u2082', viz.originX + 8, 15);
                        }

                        draw();
                        return viz;
                    }
                }
            ],

            exercises: [
                {
                    question: 'Explain why interpolation in <em>pixel space</em> (\\(\\hat{\\mathbf{x}}_\\alpha = (1-\\alpha)\\mathbf{x}_1 + \\alpha\\,\\mathbf{x}_2\\)) typically produces blurry, ghostly images, while interpolation in <em>latent space</em> (\\(g_\\theta((1-\\alpha)\\mathbf{z}_1 + \\alpha\\,\\mathbf{z}_2)\\)) can produce sharp, plausible intermediate images.',
                    hint: 'Consider what linear interpolation does geometrically in pixel space versus on the data manifold.',
                    solution: 'In pixel space, the line segment between two images generally passes <em>off</em> the data manifold: the superposition of two digit images (e.g., "3" and "7") looks like a ghostly double exposure, which is not a plausible image. In latent space, if the encoder has learned the manifold structure, the line segment stays <em>near</em> the manifold (because the manifold is locally flattened in latent coordinates). The decoder then maps these latent points back to plausible images on the manifold, producing smooth, realistic transitions.'
                },
                {
                    question: 'The manifold hypothesis says data lies near a low-dimensional manifold. How would you empirically estimate the intrinsic dimension \\(d\\) of a dataset to choose the bottleneck size?',
                    hint: 'Think about methods from manifold learning and how reconstruction error behaves as a function of \\(d\\).',
                    solution: 'Several approaches: (1) <strong>PCA explained variance</strong>: plot the fraction of variance explained vs. number of components and look for an "elbow." (2) <strong>Reconstruction error curve</strong>: train autoencoders with different \\(d\\) and plot reconstruction loss vs. \\(d\\); the curve typically shows diminishing returns past the intrinsic dimension. (3) <strong>Intrinsic dimension estimators</strong> such as the maximum likelihood estimator of Levina & Bickel (2005), which examines nearest-neighbor distances. (4) <strong>Cross-validation</strong>: choose \\(d\\) that gives the best reconstruction on held-out data (too small underfits, too large overfits).'
                },
                {
                    question: 'Consider an autoencoder with \\(d = 2\\). You encode all training images and plot them in the latent plane. You observe that digit "1" maps to a tight cluster at \\((3, 0)\\) and digit "7" maps to a tight cluster at \\((-3, 0)\\). What do you expect the decoder to produce at the midpoint \\(\\mathbf{z} = (0, 0)\\)?',
                    hint: 'Is the midpoint likely on the data manifold? Did any training data map near \\((0,0)\\)?',
                    solution: 'Since the clusters are far apart with a gap around \\((0,0)\\), no training data mapped there, so the decoder was never trained on inputs near the origin. The decoder output at \\((0,0)\\) is essentially <em>extrapolation</em> and will likely be a blurry, meaningless image, or possibly a vague blend of "1" and "7" features. This illustrates the "hole" problem: deterministic autoencoders leave gaps in the latent space where the decoder produces poor outputs. This is one of the key motivations for VAEs, which regularize the latent space to avoid such gaps.'
                }
            ]
        },

        // ===================== Section 3: Variational Autoencoders =====================
        {
            id: 'ch16-sec03',
            title: 'Variational Autoencoders',
            content: `<h2>Variational Autoencoders</h2>

                <div class="env-block intuition">
                    <div class="env-title">From Deterministic to Probabilistic</div>
                    <div class="env-body"><p>The gap problem of deterministic autoencoders stems from a fundamental issue: the encoder maps each input to a <em>single point</em> in latent space. There is no incentive to organize these points so that every region of latent space maps to a plausible output. The <strong>variational autoencoder</strong> (VAE, Kingma & Welling 2014; Rezende, Mohamed & Wierstra 2014) solves this by making the encoder <em>probabilistic</em>: instead of producing a single point \\(\\mathbf{z}\\), it produces a <em>distribution</em> over \\(\\mathbf{z}\\). A regularization term then encourages these distributions to cover the latent space uniformly.</p></div>
                </div>

                <h3>The Generative Model</h3>

                <p>A VAE defines a generative model as follows:</p>
                <ol>
                    <li>Sample a latent variable from a prior: \\(\\mathbf{z} \\sim p(\\mathbf{z}) = \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\).</li>
                    <li>Generate an observation from the likelihood: \\(\\mathbf{x} \\sim p_\\theta(\\mathbf{x} \\mid \\mathbf{z})\\).</li>
                </ol>
                <p>The decoder network parameterizes \\(p_\\theta(\\mathbf{x} \\mid \\mathbf{z})\\). For continuous data, this is typically \\(\\mathcal{N}(\\boldsymbol{\\mu}_\\theta(\\mathbf{z}), \\sigma^2 \\mathbf{I})\\), where \\(\\boldsymbol{\\mu}_\\theta\\) is the decoder network output. For binary data, it is a product of Bernoulli distributions with probabilities given by the decoder (with sigmoid output).</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (Variational Autoencoder)</div>
                    <div class="env-body"><p>A <strong>variational autoencoder</strong> consists of:</p>
                    <ol>
                        <li>A <strong>prior</strong> \\(p(\\mathbf{z}) = \\mathcal{N}(\\mathbf{0}, \\mathbf{I}_d)\\) over the latent space.</li>
                        <li>A <strong>decoder</strong> (generative network) \\(p_\\theta(\\mathbf{x} \\mid \\mathbf{z})\\) parameterized by \\(\\theta\\).</li>
                        <li>An <strong>encoder</strong> (inference/recognition network) \\(q_\\phi(\\mathbf{z} \\mid \\mathbf{x})\\) that approximates the intractable posterior \\(p_\\theta(\\mathbf{z} \\mid \\mathbf{x})\\).</li>
                    </ol>
                    <p>The encoder outputs the parameters of a Gaussian: \\(q_\\phi(\\mathbf{z} \\mid \\mathbf{x}) = \\mathcal{N}(\\boldsymbol{\\mu}_\\phi(\\mathbf{x}), \\text{diag}(\\boldsymbol{\\sigma}_\\phi^2(\\mathbf{x})))\\), where \\(\\boldsymbol{\\mu}_\\phi\\) and \\(\\log \\boldsymbol{\\sigma}_\\phi^2\\) are outputs of the encoder network.</p></div>
                </div>

                <h3>Why Is the Posterior Intractable?</h3>

                <p>We would like to compute the true posterior \\(p_\\theta(\\mathbf{z} \\mid \\mathbf{x})\\) via Bayes' rule:</p>
                \\[p_\\theta(\\mathbf{z} \\mid \\mathbf{x}) = \\frac{p_\\theta(\\mathbf{x} \\mid \\mathbf{z})\\, p(\\mathbf{z})}{p_\\theta(\\mathbf{x})}\\]
                <p>The denominator requires marginalizing over all possible latent codes:</p>
                \\[p_\\theta(\\mathbf{x}) = \\int p_\\theta(\\mathbf{x} \\mid \\mathbf{z})\\, p(\\mathbf{z})\\, d\\mathbf{z}\\]
                <p>This integral is intractable for nonlinear decoders because it involves integrating the output of a neural network over all of \\(\\mathbb{R}^d\\). The recognition network \\(q_\\phi(\\mathbf{z} \\mid \\mathbf{x})\\) is trained to approximate this posterior via <strong>amortized variational inference</strong>.</p>

                <div class="env-block theorem">
                    <div class="env-title">Proposition (KL Between Gaussians, Closed Form)</div>
                    <div class="env-body"><p>For two \\(d\\)-dimensional Gaussians \\(q = \\mathcal{N}(\\boldsymbol{\\mu}, \\text{diag}(\\boldsymbol{\\sigma}^2))\\) and \\(p = \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\):</p>
                    \\[D_{\\text{KL}}(q \\| p) = \\frac{1}{2}\\sum_{j=1}^{d} \\bigl(\\mu_j^2 + \\sigma_j^2 - \\ln \\sigma_j^2 - 1\\bigr)\\]
                    <p>This is always non-negative and equals zero iff \\(q = p\\), i.e., \\(\\boldsymbol{\\mu} = \\mathbf{0}\\) and \\(\\boldsymbol{\\sigma} = \\mathbf{1}\\).</p></div>
                </div>

                <div class="env-block proof">
                    <div class="env-title">Proof</div>
                    <div class="env-body"><p>Since both distributions are diagonal Gaussians, the KL divergence decomposes into a sum over dimensions. For the \\(j\\)-th dimension:</p>
                    \\[D_{\\text{KL}}(\\mathcal{N}(\\mu_j, \\sigma_j^2) \\| \\mathcal{N}(0, 1)) = \\int q(z_j) \\ln \\frac{q(z_j)}{p(z_j)}\\, dz_j\\]
                    <p>Expanding both Gaussian densities and simplifying:</p>
                    \\[= \\frac{1}{2}\\bigl(-\\ln \\sigma_j^2 + \\sigma_j^2 + \\mu_j^2 - 1\\bigr)\\]
                    <p>Summing over \\(j = 1, \\ldots, d\\) gives the result.</p>
                    <div class="qed">&#8718;</div></div>
                </div>

                <h3>VAE vs. AE Latent Space</h3>

                <p>The key difference is visible in the latent space. A standard AE produces scattered, irregular clusters with large gaps between them. A VAE's KL regularizer penalizes departure from \\(\\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\), pushing all encoder distributions toward the origin and encouraging overlap. The result is a smooth, continuous latent space where:</p>
                <ul>
                    <li>Every point near the origin maps to a plausible output.</li>
                    <li>Interpolation between any two points produces realistic intermediate outputs.</li>
                    <li>Sampling \\(\\mathbf{z} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\) and decoding yields meaningful new data.</li>
                </ul>

                <div class="env-block warning">
                    <div class="env-title">Warning (Posterior Collapse)</div>
                    <div class="env-body"><p>If the KL term dominates the loss, the encoder may learn to ignore the input entirely, setting \\(q_\\phi(\\mathbf{z} \\mid \\mathbf{x}) \\approx p(\\mathbf{z})\\) for all \\(\\mathbf{x}\\). This is called <strong>posterior collapse</strong> (or KL vanishing): the latent code carries no information about \\(\\mathbf{x}\\), and the decoder reduces to an unconditional generative model. Strategies to mitigate this include KL annealing (gradually increasing the KL weight during training), free bits (imposing a minimum KL per dimension), and using more expressive decoders.</p></div>
                </div>

                <div class="viz-placeholder" data-viz="viz-vae-vs-ae"></div>`,

            visualizations: [
                {
                    id: 'viz-vae-vs-ae',
                    title: 'VAE vs. AE Latent Space Comparison',
                    description: 'Left: AE latent space with scattered clusters and gaps. Right: VAE latent space with smooth, overlapping distributions centered near the origin. The VAE regularization fills the gaps.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, {width: 700, height: 380, scale: 1, originX: 0, originY: 0});
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var halfW = W / 2;
                        var panelW = halfW - 15;

                        var nClasses = 7;
                        var classColors = ['#f85149','#f0883e','#d29922','#3fb950','#3fb9a0','#58a6ff','#bc8cff'];

                        // Seed for reproducibility
                        function seededRandom(seed) {
                            var x = Math.sin(seed) * 10000;
                            return x - Math.floor(x);
                        }

                        function gaussRandom(seed) {
                            var u1 = seededRandom(seed);
                            var u2 = seededRandom(seed + 0.5);
                            if (u1 < 0.001) u1 = 0.001;
                            return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                        }

                        // Generate AE-style scattered clusters (spread apart, tight)
                        var aeClusters = [];
                        var aePoints = [];
                        for (var c = 0; c < nClasses; c++) {
                            var angle = c * Math.PI * 2 / nClasses + 0.3;
                            var r = 100 + seededRandom(c * 7 + 1) * 40;
                            var cx = panelW / 2 + r * Math.cos(angle);
                            var cy = H / 2 + r * Math.sin(angle);
                            aeClusters.push({cx: cx, cy: cy});
                            for (var i = 0; i < 35; i++) {
                                var dx = gaussRandom(c * 100 + i) * 15;
                                var dy = gaussRandom(c * 100 + i + 50) * 15;
                                aePoints.push({x: cx + dx, y: cy + dy, cls: c});
                            }
                        }

                        // Generate VAE-style overlapping clusters (near center, wider)
                        var vaePoints = [];
                        for (var c = 0; c < nClasses; c++) {
                            var angle = c * Math.PI * 2 / nClasses + 0.3;
                            var r = 35 + seededRandom(c * 13 + 2) * 25;
                            var cx = panelW / 2 + r * Math.cos(angle);
                            var cy = H / 2 + r * Math.sin(angle);
                            for (var i = 0; i < 35; i++) {
                                var dx = gaussRandom(c * 200 + i + 3) * 30;
                                var dy = gaussRandom(c * 200 + i + 53) * 30;
                                vaePoints.push({x: cx + dx, y: cy + dy, cls: c});
                            }
                        }

                        function drawPanel(offsetX, points, title, showGaussian) {
                            // Panel background
                            ctx.fillStyle = '#0f0f28';
                            ctx.fillRect(offsetX + 5, 5, panelW - 10, H - 10);
                            ctx.strokeStyle = '#2a2a5a';
                            ctx.lineWidth = 1;
                            ctx.strokeRect(offsetX + 5, 5, panelW - 10, H - 10);

                            // Draw standard normal reference for VAE
                            if (showGaussian) {
                                var gcx = offsetX + panelW / 2;
                                var gcy = H / 2;
                                for (var r = 3; r >= 1; r--) {
                                    ctx.strokeStyle = 'rgba(255,255,255,' + (0.05 + (3 - r) * 0.03) + ')';
                                    ctx.lineWidth = 1;
                                    ctx.setLineDash([4, 4]);
                                    ctx.beginPath();
                                    ctx.arc(gcx, gcy, r * 40, 0, Math.PI * 2);
                                    ctx.stroke();
                                    ctx.setLineDash([]);
                                }
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.fillStyle = '#ffffff44';
                                ctx.textAlign = 'left';
                                ctx.fillText('1\u03C3', gcx + 42, gcy - 2);
                                ctx.fillText('2\u03C3', gcx + 82, gcy - 2);
                                ctx.fillText('3\u03C3', gcx + 122, gcy - 2);
                            }

                            // Draw points
                            for (var i = 0; i < points.length; i++) {
                                var p = points[i];
                                ctx.fillStyle = classColors[p.cls] + 'aa';
                                ctx.beginPath();
                                ctx.arc(offsetX + p.x, p.y, 2.5, 0, Math.PI * 2);
                                ctx.fill();
                            }

                            // Title
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.fillStyle = '#f0f6fc';
                            ctx.textAlign = 'center';
                            ctx.fillText(title, offsetX + panelW / 2, 25);
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            drawPanel(0, aePoints, 'Standard AE Latent Space', false);
                            drawPanel(halfW, vaePoints, 'VAE Latent Space', true);

                            // Annotations
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';

                            // AE annotation: highlight a gap
                            ctx.fillStyle = '#f8514966';
                            ctx.fillText('\u2190 gaps between clusters \u2192', panelW / 2, H - 18);

                            // VAE annotation
                            ctx.fillStyle = '#3fb95066';
                            ctx.fillText('smooth, continuous coverage', halfW + panelW / 2, H - 18);

                            // Legend
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            for (var c = 0; c < nClasses; c++) {
                                var lx = W - 70;
                                var ly = 50 + c * 14;
                                ctx.fillStyle = classColors[c];
                                ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2); ctx.fill();
                                ctx.fillText('Class ' + c, lx + 8, ly + 3);
                            }
                        }

                        draw();
                        return viz;
                    }
                }
            ],

            exercises: [
                {
                    question: 'In a VAE, the encoder outputs \\(\\boldsymbol{\\mu}_\\phi(\\mathbf{x})\\) and \\(\\log \\boldsymbol{\\sigma}_\\phi^2(\\mathbf{x})\\). Why do we parameterize the log-variance \\(\\log \\sigma_j^2\\) rather than \\(\\sigma_j\\) or \\(\\sigma_j^2\\) directly?',
                    hint: 'Think about the range of a neural network output and the constraint that variance must be positive.',
                    solution: 'A neural network output is unconstrained (can be any real number). (1) If we output \\(\\sigma_j^2\\) directly, we would need to enforce positivity (e.g., via a softplus), which can cause numerical issues near zero. (2) If we output \\(\\sigma_j\\), we also need positivity. (3) Outputting \\(\\log \\sigma_j^2\\) is unconstrained: any real value maps to a valid positive variance via \\(\\sigma_j^2 = \\exp(\\log \\sigma_j^2)\\). This also provides better gradient behavior, since the KL divergence involves \\(\\log \\sigma_j^2\\) linearly, and the gradient with respect to \\(\\log \\sigma_j^2\\) is well-conditioned even when \\(\\sigma_j^2\\) is very small or very large.'
                },
                {
                    question: 'Derive the closed-form expression for \\(D_{\\text{KL}}(\\mathcal{N}(\\mu, \\sigma^2) \\| \\mathcal{N}(0, 1))\\) in one dimension.',
                    hint: 'Write out \\(\\int q(z) \\ln \\frac{q(z)}{p(z)} dz\\), expand the log-ratio using the Gaussian densities, and evaluate each resulting integral using \\(\\mathbb{E}_q[z] = \\mu\\), \\(\\mathbb{E}_q[z^2] = \\mu^2 + \\sigma^2\\).',
                    solution: 'We have \\(q(z) = \\mathcal{N}(\\mu, \\sigma^2)\\) and \\(p(z) = \\mathcal{N}(0, 1)\\). The log-ratio is: \\[\\ln \\frac{q(z)}{p(z)} = -\\frac{1}{2}\\ln(2\\pi\\sigma^2) - \\frac{(z-\\mu)^2}{2\\sigma^2} + \\frac{1}{2}\\ln(2\\pi) + \\frac{z^2}{2}\\] \\[= -\\frac{1}{2}\\ln \\sigma^2 - \\frac{(z-\\mu)^2}{2\\sigma^2} + \\frac{z^2}{2}\\] Taking the expectation under \\(q\\): \\(\\mathbb{E}_q\\left[\\frac{(z-\\mu)^2}{2\\sigma^2}\\right] = \\frac{1}{2}\\) and \\(\\mathbb{E}_q\\left[\\frac{z^2}{2}\\right] = \\frac{\\mu^2 + \\sigma^2}{2}\\). Therefore: \\[D_{\\text{KL}} = -\\frac{1}{2}\\ln \\sigma^2 - \\frac{1}{2} + \\frac{\\mu^2 + \\sigma^2}{2} = \\frac{1}{2}(\\mu^2 + \\sigma^2 - \\ln \\sigma^2 - 1)\\]'
                },
                {
                    question: 'Explain the trade-off that the VAE faces between reconstruction quality and latent space regularity. What happens at each extreme?',
                    hint: 'Consider what the loss function rewards and what each term pushes the model to do.',
                    solution: 'The VAE loss has two terms: reconstruction loss and KL divergence. (1) <strong>Reconstruction dominates</strong>: the model ignores the KL penalty, the encoder outputs very small \\(\\sigma_j\\) (point-like distributions), the latent space looks like a standard AE (separated clusters, gaps), and generation quality is poor. (2) <strong>KL dominates</strong>: the encoder is forced to output \\(q \\approx \\mathcal{N}(0, I)\\) for all inputs (posterior collapse), the latent code carries no information about the input, the decoder learns to produce a single "average" output, and reconstruction is terrible. The optimal balance yields a structured latent space that is regular enough for generation but informative enough for faithful reconstruction.'
                }
            ]
        },

        // ===================== Section 4: The Reparameterization Trick =====================
        {
            id: 'ch16-sec04',
            title: 'The Reparameterization Trick',
            content: `<h2>The Reparameterization Trick</h2>

                <div class="env-block intuition">
                    <div class="env-title">The Gradient Problem</div>
                    <div class="env-body"><p>Training a VAE requires computing gradients of the loss with respect to the encoder parameters \\(\\phi\\). But the loss involves <em>sampling</em> \\(\\mathbf{z} \\sim q_\\phi(\\mathbf{z} \\mid \\mathbf{x})\\), and we cannot backpropagate through a random sampling operation. The <strong>reparameterization trick</strong> (Kingma & Welling, 2014) is an elegant solution: instead of sampling from a distribution that depends on \\(\\phi\\), we express the sample as a <em>deterministic</em> function of \\(\\phi\\) and an <em>independent</em> random variable.</p></div>
                </div>

                <h3>The Problem: Sampling Is Not Differentiable</h3>

                <p>Consider the reconstruction term in the VAE objective (details in Section 5):</p>
                \\[\\mathcal{L}_{\\text{recon}} = \\mathbb{E}_{\\mathbf{z} \\sim q_\\phi(\\mathbf{z} \\mid \\mathbf{x})} \\bigl[ \\log p_\\theta(\\mathbf{x} \\mid \\mathbf{z}) \\bigr]\\]
                <p>To optimize this with stochastic gradient descent, we need \\(\\nabla_\\phi \\mathcal{L}_{\\text{recon}}\\). The naive approach is:</p>
                \\[\\nabla_\\phi \\mathbb{E}_{q_\\phi} [f(\\mathbf{z})] = \\nabla_\\phi \\int f(\\mathbf{z})\\, q_\\phi(\\mathbf{z} \\mid \\mathbf{x})\\, d\\mathbf{z}\\]
                <p>We cannot simply move the gradient inside the integral because \\(q_\\phi\\) depends on \\(\\phi\\). Standard Monte Carlo estimation (draw \\(\\mathbf{z} \\sim q_\\phi\\), compute \\(f(\\mathbf{z})\\)) gives a sample of the <em>expectation</em> but not of its <em>gradient</em>, because the sampling distribution itself depends on \\(\\phi\\).</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (Reparameterization Trick)</div>
                    <div class="env-body"><p>Given \\(q_\\phi(\\mathbf{z} \\mid \\mathbf{x}) = \\mathcal{N}(\\boldsymbol{\\mu}_\\phi(\\mathbf{x}), \\text{diag}(\\boldsymbol{\\sigma}_\\phi^2(\\mathbf{x})))\\), the <strong>reparameterization trick</strong> writes the sample as a deterministic transformation of a fixed-distribution random variable:</p>
                    \\[\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I}), \\qquad \\mathbf{z} = \\boldsymbol{\\mu}_\\phi(\\mathbf{x}) + \\boldsymbol{\\sigma}_\\phi(\\mathbf{x}) \\odot \\boldsymbol{\\epsilon}\\]
                    <p>where \\(\\odot\\) denotes element-wise multiplication. Now \\(\\mathbf{z}\\) is a differentiable function of \\(\\phi\\) (through \\(\\boldsymbol{\\mu}_\\phi\\) and \\(\\boldsymbol{\\sigma}_\\phi\\)), and \\(\\boldsymbol{\\epsilon}\\) does not depend on \\(\\phi\\).</p></div>
                </div>

                <div class="env-block theorem">
                    <div class="env-title">Proposition (Reparameterized Gradient)</div>
                    <div class="env-body"><p>Under the reparameterization \\(\\mathbf{z} = \\boldsymbol{\\mu}_\\phi + \\boldsymbol{\\sigma}_\\phi \\odot \\boldsymbol{\\epsilon}\\) with \\(\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\):</p>
                    \\[\\nabla_\\phi \\mathbb{E}_{q_\\phi} [f(\\mathbf{z})] = \\mathbb{E}_{\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(0,I)} \\bigl[ \\nabla_\\phi f(\\boldsymbol{\\mu}_\\phi + \\boldsymbol{\\sigma}_\\phi \\odot \\boldsymbol{\\epsilon}) \\bigr]\\]
                    <p>This can be estimated with a single Monte Carlo sample: draw \\(\\boldsymbol{\\epsilon}\\), compute \\(\\mathbf{z}\\), and backpropagate through the entire computational graph.</p></div>
                </div>

                <div class="env-block proof">
                    <div class="env-title">Proof</div>
                    <div class="env-body"><p>By the change of variables \\(\\mathbf{z} = \\boldsymbol{\\mu}_\\phi + \\boldsymbol{\\sigma}_\\phi \\odot \\boldsymbol{\\epsilon}\\), the expectation becomes:</p>
                    \\[\\mathbb{E}_{q_\\phi}[f(\\mathbf{z})] = \\int f(\\boldsymbol{\\mu}_\\phi + \\boldsymbol{\\sigma}_\\phi \\odot \\boldsymbol{\\epsilon})\\, p(\\boldsymbol{\\epsilon})\\, d\\boldsymbol{\\epsilon}\\]
                    <p>where \\(p(\\boldsymbol{\\epsilon}) = \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\) does not depend on \\(\\phi\\). Therefore:</p>
                    \\[\\nabla_\\phi \\mathbb{E}_{q_\\phi}[f(\\mathbf{z})] = \\int \\nabla_\\phi f(\\boldsymbol{\\mu}_\\phi + \\boldsymbol{\\sigma}_\\phi \\odot \\boldsymbol{\\epsilon})\\, p(\\boldsymbol{\\epsilon})\\, d\\boldsymbol{\\epsilon} = \\mathbb{E}_{\\boldsymbol{\\epsilon}}[\\nabla_\\phi f(\\boldsymbol{\\mu}_\\phi + \\boldsymbol{\\sigma}_\\phi \\odot \\boldsymbol{\\epsilon})]\\]
                    <p>The interchange of gradient and integral is valid because \\(p(\\boldsymbol{\\epsilon})\\) is independent of \\(\\phi\\), and we assume regularity conditions (Leibniz integral rule) hold.</p>
                    <div class="qed">&#8718;</div></div>
                </div>

                <h3>Computational Graph Perspective</h3>

                <p>Without reparameterization, the computational graph contains a <em>stochastic node</em> \\(\\mathbf{z} \\sim q_\\phi\\), and gradients cannot flow back through it. With reparameterization, the stochasticity is isolated in \\(\\boldsymbol{\\epsilon}\\) (an input to the graph, like data), and the path from \\(\\phi\\) to \\(\\mathbf{z}\\) to the loss is fully differentiable.</p>

                <div class="env-block remark">
                    <div class="env-title">Remark (Generality of Reparameterization)</div>
                    <div class="env-body"><p>The reparameterization trick works for any distribution that can be written as a deterministic transformation of a fixed noise source: \\(\\mathbf{z} = h_\\phi(\\boldsymbol{\\epsilon})\\). This includes Gaussians (\\(h = \\mu + \\sigma \\cdot \\epsilon\\)), log-normals, exponentials, and many others. It does <em>not</em> directly apply to discrete distributions (e.g., Categorical), which require alternative techniques like the Gumbel-Softmax trick (Jang et al., 2017; Maddison et al., 2017) or REINFORCE-style score function estimators.</p></div>
                </div>

                <div class="env-block example">
                    <div class="env-title">Example (Single-Sample Gradient Estimate)</div>
                    <div class="env-body"><p>In practice, a single sample \\(\\boldsymbol{\\epsilon}^{(1)} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\) per data point per training step is sufficient. The gradient estimate is:</p>
                    \\[\\hat{g} = \\nabla_\\phi \\bigl[ \\log p_\\theta(\\mathbf{x} \\mid \\boldsymbol{\\mu}_\\phi + \\boldsymbol{\\sigma}_\\phi \\odot \\boldsymbol{\\epsilon}^{(1)}) \\bigr]\\]
                    <p>This is an unbiased estimate of the true gradient. Despite using only one sample, it works remarkably well in practice when combined with minibatch averaging over data points.</p></div>
                </div>

                <div class="viz-placeholder" data-viz="viz-reparam-trick"></div>`,

            visualizations: [
                {
                    id: 'viz-reparam-trick',
                    title: 'The Reparameterization Trick',
                    description: 'Left: without reparameterization, sampling blocks gradient flow. Right: with reparameterization, randomness is separated into an external input, making the path differentiable. Adjust \\(\\mu\\) and \\(\\sigma\\) to see how the sampled z changes.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, {width: 700, height: 420, scale: 1, originX: 0, originY: 0});
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        var mu = 1.0;
                        var sigma = 0.5;
                        var epsilon = 0;
                        var animT = 0;

                        VizEngine.createSlider(controls, '\u03BC', -3, 3, mu, 0.1, function(v) { mu = v; });
                        VizEngine.createSlider(controls, '\u03C3', 0.1, 2, sigma, 0.1, function(v) { sigma = v; });
                        VizEngine.createButton(controls, 'Sample \u03B5', function() {
                            // Box-Muller
                            var u1 = Math.random(), u2 = Math.random();
                            epsilon = Math.sqrt(-2 * Math.log(u1 < 0.001 ? 0.001 : u1)) * Math.cos(2 * Math.PI * u2);
                        });

                        // Draw a box node
                        function drawBox(x, y, w, h, label, color, filled) {
                            ctx.strokeStyle = color;
                            ctx.lineWidth = 2;
                            if (filled) {
                                ctx.fillStyle = color + '22';
                                ctx.fillRect(x - w/2, y - h/2, w, h);
                            }
                            ctx.strokeRect(x - w/2, y - h/2, w, h);
                            ctx.fillStyle = color;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText(label, x, y);
                        }

                        // Draw an arrow between two points
                        function drawArrow(x1, y1, x2, y2, color, dashed, label) {
                            var dx = x2 - x1, dy = y2 - y1;
                            var len = Math.sqrt(dx*dx + dy*dy);
                            if (len < 1) return;
                            var ux = dx/len, uy = dy/len;

                            ctx.strokeStyle = color;
                            ctx.lineWidth = 2;
                            if (dashed) ctx.setLineDash([5, 4]);
                            ctx.beginPath();
                            ctx.moveTo(x1, y1);
                            ctx.lineTo(x2 - ux * 8, y2 - uy * 8);
                            ctx.stroke();
                            if (dashed) ctx.setLineDash([]);

                            // Arrowhead
                            var angle = Math.atan2(dy, dx);
                            ctx.fillStyle = color;
                            ctx.beginPath();
                            ctx.moveTo(x2, y2);
                            ctx.lineTo(x2 - 10 * Math.cos(angle - Math.PI/6), y2 - 10 * Math.sin(angle - Math.PI/6));
                            ctx.lineTo(x2 - 10 * Math.cos(angle + Math.PI/6), y2 - 10 * Math.sin(angle + Math.PI/6));
                            ctx.closePath();
                            ctx.fill();

                            if (label) {
                                var mx = (x1 + x2) / 2;
                                var my = (y1 + y2) / 2 - 10;
                                ctx.fillStyle = color;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'bottom';
                                ctx.fillText(label, mx, my);
                            }
                        }

                        // Draw X symbol (blocked gradient)
                        function drawBlock(x, y, color) {
                            ctx.strokeStyle = color;
                            ctx.lineWidth = 3;
                            var s = 8;
                            ctx.beginPath(); ctx.moveTo(x-s, y-s); ctx.lineTo(x+s, y+s); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(x+s, y-s); ctx.lineTo(x-s, y+s); ctx.stroke();
                        }

                        function draw(t) {
                            animT = t / 1000;
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var midX = W / 2;
                            var dividerX = midX;

                            // Divider
                            ctx.strokeStyle = '#2a2a5a';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([4, 4]);
                            ctx.beginPath(); ctx.moveTo(dividerX, 0); ctx.lineTo(dividerX, H); ctx.stroke();
                            ctx.setLineDash([]);

                            // ===== LEFT PANEL: Without reparameterization =====
                            var lc = midX / 2;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.red;
                            ctx.textAlign = 'center';
                            ctx.fillText('Without Reparameterization', lc, 25);

                            // Nodes
                            var lx_x = lc - 80, ly_x = 100;
                            var lx_phi = lc + 80, ly_phi = 100;
                            var lx_q = lc, ly_q = 190;
                            var lx_z = lc, ly_z = 270;
                            var lx_loss = lc, ly_loss = 350;

                            drawBox(lx_x, ly_x, 50, 30, 'x', viz.colors.blue, true);
                            drawBox(lx_phi, ly_phi, 50, 30, '\u03D5', viz.colors.teal, true);
                            drawBox(lx_q, ly_q, 80, 30, 'q\u03D5(z|x)', viz.colors.purple, true);
                            drawBox(lx_loss, ly_loss, 70, 30, 'Loss', viz.colors.orange, true);

                            // z is a stochastic node
                            ctx.fillStyle = viz.colors.red + '22';
                            ctx.beginPath(); ctx.arc(lx_z, ly_z, 22, 0, Math.PI * 2); ctx.fill();
                            ctx.strokeStyle = viz.colors.red;
                            ctx.lineWidth = 2;
                            ctx.beginPath(); ctx.arc(lx_z, ly_z, 22, 0, Math.PI * 2); ctx.stroke();
                            ctx.fillStyle = viz.colors.red;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                            ctx.fillText('z ~ q', lx_z, ly_z);

                            // Forward arrows
                            drawArrow(lx_x, ly_x + 15, lx_q - 20, ly_q - 15, viz.colors.text, false, '');
                            drawArrow(lx_phi, ly_phi + 15, lx_q + 20, ly_q - 15, viz.colors.text, false, '');
                            drawArrow(lx_q, ly_q + 15, lx_z, ly_z - 22, viz.colors.text, false, 'sample');
                            drawArrow(lx_z, ly_z + 22, lx_loss, ly_loss - 15, viz.colors.text, false, '');

                            // Blocked gradient
                            var bgx = lx_z, bgy = ly_z - 22 - 15;
                            drawBlock(lx_q, ly_q + 40, viz.colors.red);
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.red;
                            ctx.fillText('\u2207\u03D5 blocked!', lx_q + 50, ly_q + 42);

                            // ===== RIGHT PANEL: With reparameterization =====
                            var rc = midX + midX / 2;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.green;
                            ctx.textAlign = 'center';
                            ctx.fillText('With Reparameterization', rc, 25);

                            var rx_x = rc - 100, ry_x = 80;
                            var rx_phi = rc - 100, ry_phi = 140;
                            var rx_enc = rc - 30, ry_enc = 110;
                            var rx_mu = rc - 60, ry_mu = 190;
                            var rx_sig = rc + 0, ry_sig = 190;
                            var rx_eps = rc + 100, ry_eps = 190;
                            var rx_z = rc + 20, ry_z = 270;
                            var rx_loss = rc, ry_loss = 350;

                            drawBox(rx_x, ry_x, 40, 26, 'x', viz.colors.blue, true);
                            drawBox(rx_phi, ry_phi, 40, 26, '\u03D5', viz.colors.teal, true);
                            drawBox(rx_enc, ry_enc, 70, 30, 'Encoder', viz.colors.purple, true);

                            // mu and sigma outputs
                            drawBox(rx_mu, ry_mu, 40, 26, '\u03BC', viz.colors.blue, true);
                            drawBox(rx_sig, ry_sig, 40, 26, '\u03C3', viz.colors.blue, true);

                            // Epsilon - external noise
                            ctx.fillStyle = viz.colors.yellow + '22';
                            ctx.beginPath(); ctx.arc(rx_eps, ry_eps, 20, 0, Math.PI * 2); ctx.fill();
                            ctx.strokeStyle = viz.colors.yellow;
                            ctx.lineWidth = 2;
                            ctx.setLineDash([3, 3]);
                            ctx.beginPath(); ctx.arc(rx_eps, ry_eps, 20, 0, Math.PI * 2); ctx.stroke();
                            ctx.setLineDash([]);
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                            ctx.fillText('\u03B5~N(0,I)', rx_eps, ry_eps);

                            // z = mu + sigma * epsilon (deterministic node)
                            var zVal = mu + sigma * epsilon;
                            drawBox(rx_z, ry_z, 90, 30, 'z = \u03BC + \u03C3\u00B7\u03B5', viz.colors.green, true);
                            drawBox(rx_loss, ry_loss, 70, 30, 'Loss', viz.colors.orange, true);

                            // Arrows
                            drawArrow(rx_x, ry_x + 13, rx_enc - 20, ry_enc - 10, viz.colors.text, false, '');
                            drawArrow(rx_phi, ry_phi - 13, rx_enc - 20, ry_enc + 10, viz.colors.text, false, '');
                            drawArrow(rx_enc - 10, ry_enc + 15, rx_mu, ry_mu - 13, viz.colors.text, false, '');
                            drawArrow(rx_enc + 10, ry_enc + 15, rx_sig, ry_sig - 13, viz.colors.text, false, '');
                            drawArrow(rx_mu, ry_mu + 13, rx_z - 20, ry_z - 15, viz.colors.blue, false, '');
                            drawArrow(rx_sig, ry_sig + 13, rx_z, ry_z - 15, viz.colors.blue, false, '');
                            drawArrow(rx_eps, ry_eps + 20, rx_z + 20, ry_z - 15, viz.colors.yellow, true, '');
                            drawArrow(rx_z, ry_z + 15, rx_loss, ry_loss - 15, viz.colors.text, false, '');

                            // Gradient flow arrow (green, going up)
                            var gradPulse = 0.5 + 0.5 * Math.sin(animT * 3);
                            ctx.globalAlpha = 0.3 + 0.4 * gradPulse;
                            drawArrow(rx_loss - 30, ry_loss - 20, rx_z - 30, ry_z + 20, viz.colors.green, false, '');
                            drawArrow(rx_z - 35, ry_z - 20, rx_mu - 15, ry_mu + 20, viz.colors.green, false, '');
                            ctx.globalAlpha = 1;

                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.green;
                            ctx.textAlign = 'right';
                            ctx.fillText('\u2207\u03D5 flows!', rx_loss - 35, (ry_loss + ry_z) / 2);

                            // Current values display
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.text;
                            ctx.textAlign = 'left';
                            ctx.fillText('\u03BC = ' + mu.toFixed(1) + '  \u03C3 = ' + sigma.toFixed(1) + '  \u03B5 = ' + epsilon.toFixed(2) + '  z = ' + zVal.toFixed(2), midX + 15, H - 12);
                        }

                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Show that if \\(\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\) and \\(\\mathbf{z} = \\boldsymbol{\\mu} + \\boldsymbol{\\sigma} \\odot \\boldsymbol{\\epsilon}\\), then \\(\\mathbf{z} \\sim \\mathcal{N}(\\boldsymbol{\\mu}, \\text{diag}(\\boldsymbol{\\sigma}^2))\\).',
                    hint: 'Use the fact that an affine transformation of a Gaussian is Gaussian, and compute the mean and covariance.',
                    solution: 'Since \\(\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\), each component \\(\\epsilon_j \\sim \\mathcal{N}(0, 1)\\) independently. The transformation \\(z_j = \\mu_j + \\sigma_j \\epsilon_j\\) is affine in \\(\\epsilon_j\\), so \\(z_j\\) is Gaussian. We compute: \\(\\mathbb{E}[z_j] = \\mu_j + \\sigma_j \\cdot 0 = \\mu_j\\), and \\(\\text{Var}(z_j) = \\sigma_j^2 \\text{Var}(\\epsilon_j) = \\sigma_j^2\\). Since the \\(\\epsilon_j\\) are independent, so are the \\(z_j\\). Therefore \\(\\mathbf{z} \\sim \\mathcal{N}(\\boldsymbol{\\mu}, \\text{diag}(\\boldsymbol{\\sigma}^2))\\), as desired.'
                },
                {
                    question: 'Why does the reparameterization trick not apply directly to discrete latent variables (e.g., \\(\\mathbf{z} \\in \\{0, 1\\}^d\\))? What alternative approaches exist?',
                    hint: 'Think about whether a discrete random variable can be written as a continuous, differentiable function of a noise source.',
                    solution: 'Discrete random variables cannot be expressed as a <em>differentiable</em> function of continuous noise: any such function would involve a step/threshold, which has zero gradient almost everywhere. Therefore, the standard chain rule cannot propagate gradients through the sampling. Alternatives: (1) <strong>Gumbel-Softmax</strong> (Concrete distribution): relax the discrete variable to a continuous approximation using the Gumbel-max trick with a temperature parameter that approaches zero. (2) <strong>REINFORCE</strong> (score function estimator): \\(\\nabla_\\phi \\mathbb{E}_{q_\\phi}[f(z)] = \\mathbb{E}_{q_\\phi}[f(z) \\nabla_\\phi \\log q_\\phi(z)]\\), which is unbiased but has high variance. (3) <strong>Straight-through estimator</strong>: use the discrete sample in the forward pass but approximate the gradient as if the rounding operation were the identity.'
                },
                {
                    question: 'In practice, VAEs use a single sample \\(\\boldsymbol{\\epsilon}\\) per data point per gradient step. Explain why this gives a valid (if noisy) gradient estimate and why it works despite the high variance.',
                    hint: 'Consider the unbiasedness of the estimator and the role of minibatch averaging.',
                    solution: 'The reparameterized gradient \\(\\nabla_\\phi f(\\boldsymbol{\\mu}_\\phi + \\boldsymbol{\\sigma}_\\phi \\odot \\boldsymbol{\\epsilon})\\) with a single \\(\\boldsymbol{\\epsilon}\\) is an <em>unbiased</em> estimate of \\(\\mathbb{E}_{\\boldsymbol{\\epsilon}}[\\nabla_\\phi f(\\cdot)]\\). Unbiased SGD converges under standard conditions (Robbins-Monro) regardless of variance, just more slowly. In practice, variance is controlled by: (1) minibatch averaging over \\(N\\) data points (each with its own \\(\\boldsymbol{\\epsilon}\\)), reducing variance by \\(1/N\\); (2) the reparameterization gradient has much lower variance than REINFORCE-style alternatives because it uses the structure of the function \\(f\\) (gradients through the decoder network); (3) adaptive optimizers like Adam further smooth out noise.'
                }
            ]
        },

        // ===================== Section 5: ELBO & Training =====================
        {
            id: 'ch16-sec05',
            title: 'ELBO & Training',
            content: `<h2>ELBO & Training</h2>

                <div class="env-block intuition">
                    <div class="env-title">The Evidence Lower Bound</div>
                    <div class="env-body"><p>We want to maximize the log-likelihood \\(\\log p_\\theta(\\mathbf{x})\\) of the observed data, but this is intractable due to the integral over \\(\\mathbf{z}\\). Instead, we maximize a tractable lower bound called the <strong>ELBO</strong> (Evidence Lower BOund). The ELBO decomposes into two interpretable terms: one measuring reconstruction quality, the other measuring how close the approximate posterior is to the prior.</p></div>
                </div>

                <h3>Deriving the ELBO</h3>

                <p>Starting from the log-marginal likelihood:</p>
                \\[\\log p_\\theta(\\mathbf{x}) = \\log \\int p_\\theta(\\mathbf{x} \\mid \\mathbf{z})\\, p(\\mathbf{z})\\, d\\mathbf{z}\\]

                <p>We introduce the approximate posterior \\(q_\\phi(\\mathbf{z} \\mid \\mathbf{x})\\) and apply Jensen's inequality:</p>
                \\[\\log p_\\theta(\\mathbf{x}) = \\log \\int \\frac{q_\\phi(\\mathbf{z} \\mid \\mathbf{x})}{q_\\phi(\\mathbf{z} \\mid \\mathbf{x})} p_\\theta(\\mathbf{x} \\mid \\mathbf{z})\\, p(\\mathbf{z})\\, d\\mathbf{z} \\geq \\int q_\\phi(\\mathbf{z} \\mid \\mathbf{x}) \\log \\frac{p_\\theta(\\mathbf{x} \\mid \\mathbf{z})\\, p(\\mathbf{z})}{q_\\phi(\\mathbf{z} \\mid \\mathbf{x})}\\, d\\mathbf{z}\\]

                <div class="env-block theorem">
                    <div class="env-title">Theorem (ELBO Decomposition)</div>
                    <div class="env-body"><p>The <strong>Evidence Lower Bound</strong> satisfies:</p>
                    \\[\\log p_\\theta(\\mathbf{x}) \\geq \\underbrace{\\mathbb{E}_{q_\\phi(\\mathbf{z} \\mid \\mathbf{x})}[\\log p_\\theta(\\mathbf{x} \\mid \\mathbf{z})]}_{{\\text{Reconstruction term}}} - \\underbrace{D_{\\text{KL}}(q_\\phi(\\mathbf{z} \\mid \\mathbf{x}) \\| p(\\mathbf{z}))}_{{\\text{KL regularization term}}} = \\text{ELBO}(\\phi, \\theta; \\mathbf{x})\\]
                    <p>Equality holds iff \\(q_\\phi(\\mathbf{z} \\mid \\mathbf{x}) = p_\\theta(\\mathbf{z} \\mid \\mathbf{x})\\), i.e., the approximate posterior matches the true posterior exactly.</p></div>
                </div>

                <div class="env-block proof">
                    <div class="env-title">Proof</div>
                    <div class="env-body"><p>An alternative (and more illuminating) derivation avoids Jensen's inequality. Start with:</p>
                    \\[\\log p_\\theta(\\mathbf{x}) = \\mathbb{E}_{q_\\phi(\\mathbf{z} \\mid \\mathbf{x})}[\\log p_\\theta(\\mathbf{x})]\\]
                    <p>since \\(\\log p_\\theta(\\mathbf{x})\\) does not depend on \\(\\mathbf{z}\\). Now apply Bayes' rule \\(p_\\theta(\\mathbf{x}) = \\frac{p_\\theta(\\mathbf{x} \\mid \\mathbf{z}) p(\\mathbf{z})}{p_\\theta(\\mathbf{z} \\mid \\mathbf{x})}\\):</p>
                    \\[= \\mathbb{E}_{q_\\phi}\\left[\\log \\frac{p_\\theta(\\mathbf{x} \\mid \\mathbf{z})\\, p(\\mathbf{z})}{p_\\theta(\\mathbf{z} \\mid \\mathbf{x})}\\right]\\]
                    \\[= \\mathbb{E}_{q_\\phi}\\left[\\log \\frac{p_\\theta(\\mathbf{x} \\mid \\mathbf{z})\\, p(\\mathbf{z})}{q_\\phi(\\mathbf{z} \\mid \\mathbf{x})} \\cdot \\frac{q_\\phi(\\mathbf{z} \\mid \\mathbf{x})}{p_\\theta(\\mathbf{z} \\mid \\mathbf{x})}\\right]\\]
                    \\[= \\underbrace{\\mathbb{E}_{q_\\phi}\\left[\\log \\frac{p_\\theta(\\mathbf{x} \\mid \\mathbf{z})\\, p(\\mathbf{z})}{q_\\phi(\\mathbf{z} \\mid \\mathbf{x})}\\right]}_{\\text{ELBO}} + \\underbrace{D_{\\text{KL}}(q_\\phi(\\mathbf{z} \\mid \\mathbf{x}) \\| p_\\theta(\\mathbf{z} \\mid \\mathbf{x}))}_{\\geq 0}\\]
                    <p>Since KL divergence is non-negative, \\(\\log p_\\theta(\\mathbf{x}) \\geq \\text{ELBO}\\). Equality holds iff \\(q_\\phi = p_\\theta(\\cdot \\mid \\mathbf{x})\\).</p>
                    <div class="qed">&#8718;</div></div>
                </div>

                <h3>Interpreting the Two Terms</h3>

                <p>The ELBO has an elegant decomposition:</p>
                <ul>
                    <li><strong>Reconstruction term</strong> \\(\\mathbb{E}_{q_\\phi}[\\log p_\\theta(\\mathbf{x} \\mid \\mathbf{z})]\\): encourages the decoder to faithfully reconstruct the input. For Gaussian \\(p_\\theta(\\mathbf{x} \\mid \\mathbf{z})\\), this is (up to constants) the negative MSE. For Bernoulli, it is the negative binary cross-entropy.</li>
                    <li><strong>KL term</strong> \\(D_{\\text{KL}}(q_\\phi(\\mathbf{z} \\mid \\mathbf{x}) \\| p(\\mathbf{z}))\\): regularizes the latent space by penalizing deviation of the approximate posterior from the standard Gaussian prior. This prevents the encoder from placing all mass on isolated points and encourages a smooth latent space.</li>
                </ul>

                <div class="env-block definition">
                    <div class="env-title">Definition (\\(\\beta\\)-VAE)</div>
                    <div class="env-body"><p>The <strong>\\(\\beta\\)-VAE</strong> (Higgins et al., 2017) modifies the ELBO by weighting the KL term:</p>
                    \\[\\mathcal{L}_{\\beta\\text{-VAE}} = \\mathbb{E}_{q_\\phi}[\\log p_\\theta(\\mathbf{x} \\mid \\mathbf{z})] - \\beta \\cdot D_{\\text{KL}}(q_\\phi(\\mathbf{z} \\mid \\mathbf{x}) \\| p(\\mathbf{z}))\\]
                    <p>When \\(\\beta = 1\\), this is the standard ELBO. When \\(\\beta &gt; 1\\), the latent space is more strongly regularized, encouraging disentangled representations (each latent dimension captures an independent factor of variation). When \\(\\beta &lt; 1\\), reconstruction quality improves at the cost of less regular latent structure.</p></div>
                </div>

                <h3>The Full Training Algorithm</h3>

                <p>Putting it all together, each training step of a VAE proceeds as follows:</p>
                <ol>
                    <li>Sample a minibatch \\(\\{\\mathbf{x}^{(1)}, \\ldots, \\mathbf{x}^{(M)}\\}\\).</li>
                    <li>For each \\(\\mathbf{x}^{(i)}\\), run the encoder to get \\(\\boldsymbol{\\mu}^{(i)} = \\boldsymbol{\\mu}_\\phi(\\mathbf{x}^{(i)})\\), \\(\\log \\boldsymbol{\\sigma}^{2(i)} = \\log \\boldsymbol{\\sigma}^2_\\phi(\\mathbf{x}^{(i)})\\).</li>
                    <li>Sample \\(\\boldsymbol{\\epsilon}^{(i)} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\) and compute \\(\\mathbf{z}^{(i)} = \\boldsymbol{\\mu}^{(i)} + \\boldsymbol{\\sigma}^{(i)} \\odot \\boldsymbol{\\epsilon}^{(i)}\\).</li>
                    <li>Run the decoder to get \\(\\hat{\\mathbf{x}}^{(i)} = g_\\theta(\\mathbf{z}^{(i)})\\).</li>
                    <li>Compute the loss: \\(\\mathcal{L} = -\\frac{1}{M}\\sum_{i=1}^{M}\\bigl[\\log p_\\theta(\\mathbf{x}^{(i)} \\mid \\mathbf{z}^{(i)}) - \\beta \\cdot D_{\\text{KL}}(q_\\phi(\\mathbf{z} \\mid \\mathbf{x}^{(i)}) \\| p(\\mathbf{z}))\\bigr]\\).</li>
                    <li>Backpropagate and update \\(\\phi, \\theta\\) via Adam (or similar).</li>
                </ol>

                <div class="env-block remark">
                    <div class="env-title">Remark (Connection to Regularized Autoencoders)</div>
                    <div class="env-body"><p>When \\(\\beta = 0\\), the VAE reduces to a standard autoencoder with no latent regularization. When \\(\\beta \\to \\infty\\), the encoder is forced to match the prior exactly, and the model degenerates. The intermediate \\(\\beta\\) values trace a smooth frontier between reconstruction fidelity and generative quality. This perspective connects VAEs to the broader framework of <strong>rate-distortion theory</strong> (from information theory), where the reconstruction term is the distortion and the KL term is the rate.</p></div>
                </div>

                <div class="env-block example">
                    <div class="env-title">Example (VAE on MNIST)</div>
                    <div class="env-body"><p>A standard VAE on MNIST with \\(d = 2\\), two-layer MLP encoder/decoder (hidden size 256), sigmoid decoder output, BCE reconstruction loss, and \\(\\beta = 1\\) typically achieves:</p>
                    <ul>
                        <li>Reconstruction loss: roughly 100-120 nats (sum over 784 pixels).</li>
                        <li>KL divergence: roughly 5-15 nats (summed over 2 latent dims).</li>
                        <li>Generated samples from \\(\\mathbf{z} \\sim \\mathcal{N}(0, \\mathbf{I})\\) look like blurry but recognizable digits.</li>
                        <li>Latent space shows smooth transitions between digit classes.</li>
                    </ul>
                    <p>Blurriness is a well-known limitation of VAEs, stemming from the Gaussian decoder assumption: the model averages over possible outputs rather than committing to a sharp reconstruction.</p></div>
                </div>

                <div class="viz-placeholder" data-viz="viz-elbo-decomp"></div>`,

            visualizations: [
                {
                    id: 'viz-elbo-decomp',
                    title: 'ELBO Decomposition & \\(\\beta\\)-VAE Trade-off',
                    description: 'Adjust \\(\\beta\\) to see how the balance between reconstruction loss and KL divergence changes. Higher \\(\\beta\\) forces the latent space to be more regular (lower KL) but sacrifices reconstruction quality. The right panel shows the effect on latent space structure.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, {width: 700, height: 420, scale: 1, originX: 0, originY: 0});
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        var beta = 1.0;
                        var animT = 0;

                        var betaSlider = VizEngine.createSlider(controls, '\u03B2', 0, 4, 1, 0.1, function(v) { beta = v; });

                        // Simulate how recon and KL change with beta
                        // In reality: higher beta -> lower KL, higher recon loss
                        function getRecon(b) {
                            // Sigmoid-like curve: recon increases with beta
                            return 30 + 70 * (1 - Math.exp(-0.5 * b));
                        }
                        function getKL(b) {
                            // KL decreases with beta
                            return Math.max(0.5, 15 * Math.exp(-0.8 * b));
                        }

                        function draw(t) {
                            animT = t / 1000;
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var recon = getRecon(beta);
                            var kl = getKL(beta);
                            var elbo = -(recon + beta * kl);

                            // ===== LEFT: Bar chart of ELBO decomposition =====
                            var barAreaX = 40;
                            var barAreaW = 260;
                            var barAreaY = 60;
                            var barAreaH = 280;
                            var barBottom = barAreaY + barAreaH;

                            // Title
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.white;
                            ctx.textAlign = 'center';
                            ctx.fillText('ELBO Decomposition', barAreaX + barAreaW / 2, 30);

                            // Y-axis scale
                            var maxVal = 150;
                            var scaleY = barAreaH / maxVal;

                            // Draw y-axis
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(barAreaX, barAreaY);
                            ctx.lineTo(barAreaX, barBottom);
                            ctx.lineTo(barAreaX + barAreaW, barBottom);
                            ctx.stroke();

                            // Y-axis labels
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.text;
                            ctx.textAlign = 'right';
                            for (var v = 0; v <= maxVal; v += 30) {
                                var yy = barBottom - v * scaleY;
                                ctx.fillText(v.toString(), barAreaX - 5, yy + 3);
                                ctx.strokeStyle = viz.colors.grid;
                                ctx.beginPath(); ctx.moveTo(barAreaX, yy); ctx.lineTo(barAreaX + barAreaW, yy); ctx.stroke();
                            }

                            // Stacked bar for total loss
                            var barW = 60;
                            var bar1X = barAreaX + 30;
                            var reconH = recon * scaleY;
                            var klH = beta * kl * scaleY;

                            // Reconstruction (bottom)
                            ctx.fillStyle = viz.colors.blue;
                            ctx.fillRect(bar1X, barBottom - reconH, barW, reconH);
                            ctx.strokeStyle = viz.colors.blue;
                            ctx.lineWidth = 1;
                            ctx.strokeRect(bar1X, barBottom - reconH, barW, reconH);

                            // KL (top, stacked)
                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillRect(bar1X, barBottom - reconH - klH, barW, klH);
                            ctx.strokeStyle = viz.colors.orange;
                            ctx.strokeRect(bar1X, barBottom - reconH - klH, barW, klH);

                            // Labels on bars
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            if (reconH > 20) {
                                ctx.fillStyle = '#fff';
                                ctx.fillText('Recon', bar1X + barW/2, barBottom - reconH/2 - 6);
                                ctx.fillText(recon.toFixed(1), bar1X + barW/2, barBottom - reconH/2 + 8);
                            }
                            if (klH > 15) {
                                ctx.fillStyle = '#fff';
                                ctx.fillText('\u03B2\u00B7KL', bar1X + barW/2, barBottom - reconH - klH/2 - 6);
                                ctx.fillText((beta * kl).toFixed(1), bar1X + barW/2, barBottom - reconH - klH/2 + 8);
                            }

                            // Separate bars for Recon and KL
                            var bar2X = bar1X + barW + 30;
                            var bar3X = bar2X + barW + 20;

                            ctx.fillStyle = viz.colors.blue + '88';
                            ctx.fillRect(bar2X, barBottom - reconH, barW, reconH);
                            ctx.strokeStyle = viz.colors.blue;
                            ctx.strokeRect(bar2X, barBottom - reconH, barW, reconH);

                            ctx.fillStyle = viz.colors.orange + '88';
                            var rawKLH = kl * scaleY;
                            ctx.fillRect(bar3X, barBottom - rawKLH, barW, rawKLH);
                            ctx.strokeStyle = viz.colors.orange;
                            ctx.strokeRect(bar3X, barBottom - rawKLH, barW, rawKLH);

                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.text;
                            ctx.textAlign = 'center';
                            ctx.fillText('Total', bar1X + barW/2, barBottom + 16);
                            ctx.fillText('-Recon', bar2X + barW/2, barBottom + 16);
                            ctx.fillText('KL', bar3X + barW/2, barBottom + 16);

                            // ELBO value
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.green;
                            ctx.textAlign = 'center';
                            ctx.fillText('ELBO \u2248 ' + elbo.toFixed(1), barAreaX + barAreaW/2, barBottom + 38);

                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('ELBO = -Recon - \u03B2\u00B7KL', barAreaX + barAreaW/2, barBottom + 55);

                            // ===== RIGHT: Latent space effect =====
                            var rPanelX = 360;
                            var rPanelW = 310;
                            var rPanelY = 50;
                            var rPanelH = 310;
                            var rcx = rPanelX + rPanelW / 2;
                            var rcy = rPanelY + rPanelH / 2;

                            // Panel border
                            ctx.strokeStyle = '#2a2a5a';
                            ctx.lineWidth = 1;
                            ctx.strokeRect(rPanelX, rPanelY, rPanelW, rPanelH);

                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.white;
                            ctx.textAlign = 'center';
                            ctx.fillText('Latent Space (effect of \u03B2)', rcx, 30);

                            // Draw N(0,I) reference
                            ctx.strokeStyle = '#ffffff22';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([3, 3]);
                            for (var r = 1; r <= 3; r++) {
                                ctx.beginPath(); ctx.arc(rcx, rcy, r * 35, 0, Math.PI * 2); ctx.stroke();
                            }
                            ctx.setLineDash([]);

                            // Generate clusters based on beta
                            // Higher beta -> more overlap, closer to center
                            var nCls = 5;
                            var clsColors = ['#f85149','#f0883e','#3fb950','#58a6ff','#bc8cff'];

                            function seeded(s) { var x = Math.sin(s * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); }

                            for (var c = 0; c < nCls; c++) {
                                var angle = c * Math.PI * 2 / nCls + 0.5;
                                // Cluster distance from center: smaller with higher beta
                                var dist = (3.5 - beta * 0.7) * 35;
                                if (dist < 10) dist = 10;
                                var ccx = rcx + dist * Math.cos(angle);
                                var ccy = rcy + dist * Math.sin(angle);
                                // Cluster spread: larger with higher beta (more overlap)
                                var spread = 8 + beta * 6;

                                for (var i = 0; i < 25; i++) {
                                    var u1 = seeded(c * 100 + i * 2.71);
                                    var u2 = seeded(c * 100 + i * 3.14 + 0.5);
                                    if (u1 < 0.001) u1 = 0.001;
                                    var g1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                                    var g2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
                                    var px = ccx + g1 * spread;
                                    var py = ccy + g2 * spread;

                                    // Clip to panel
                                    if (px > rPanelX + 5 && px < rPanelX + rPanelW - 5 &&
                                        py > rPanelY + 5 && py < rPanelY + rPanelH - 5) {
                                        ctx.fillStyle = clsColors[c] + 'aa';
                                        ctx.beginPath(); ctx.arc(px, py, 2.5, 0, Math.PI * 2); ctx.fill();
                                    }
                                }
                            }

                            // Annotation
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.text;
                            ctx.textAlign = 'center';
                            if (beta < 0.3) {
                                ctx.fillText('Low \u03B2: separated clusters, gaps', rcx, rPanelY + rPanelH - 8);
                            } else if (beta > 2.5) {
                                ctx.fillText('High \u03B2: collapsed near prior', rcx, rPanelY + rPanelH - 8);
                            } else {
                                ctx.fillText('\u03B2 = ' + beta.toFixed(1) + ': balanced structure', rcx, rPanelY + rPanelH - 8);
                            }

                            // Legend
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.blue;
                            ctx.textAlign = 'left';
                            ctx.fillText('\u25A0 Reconstruction loss', rPanelX, H - 20);
                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillText('\u25A0 KL divergence', rPanelX + 160, H - 20);
                        }

                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Starting from \\(\\log p_\\theta(\\mathbf{x}) = \\text{ELBO} + D_{\\text{KL}}(q_\\phi \\| p_\\theta(\\cdot \\mid \\mathbf{x}))\\), explain why maximizing the ELBO with respect to \\(\\phi\\) simultaneously tightens the bound and improves the approximate posterior.',
                    hint: 'Consider what happens to each term when \\(\\phi\\) changes. Note that \\(\\log p_\\theta(\\mathbf{x})\\) does not depend on \\(\\phi\\).',
                    solution: 'Since \\(\\log p_\\theta(\\mathbf{x})\\) is constant with respect to \\(\\phi\\), we have \\(\\text{ELBO} = \\log p_\\theta(\\mathbf{x}) - D_{\\text{KL}}(q_\\phi \\| p_\\theta(\\cdot \\mid \\mathbf{x}))\\). Increasing the ELBO (with respect to \\(\\phi\\)) necessarily <em>decreases</em> \\(D_{\\text{KL}}(q_\\phi \\| p_\\theta(\\cdot \\mid \\mathbf{x}))\\), meaning the approximate posterior \\(q_\\phi\\) moves closer to the true posterior. Simultaneously, the gap between \\(\\log p_\\theta(\\mathbf{x})\\) and the ELBO shrinks, so the bound becomes tighter. With respect to \\(\\theta\\), maximizing the ELBO increases the model evidence \\(\\log p_\\theta(\\mathbf{x})\\) (plus potentially changes the gap). Thus the ELBO serves as a unified objective for both inference (\\(\\phi\\)) and learning (\\(\\theta\\)).'
                },
                {
                    question: 'In a \\(\\beta\\)-VAE with \\(\\beta = 4\\), suppose the reconstruction loss is 100 and the KL divergence is 10. Compute the total loss. Then explain what would happen to the generated samples if we increased \\(\\beta\\) further.',
                    hint: 'The loss is \\(-\\text{ELBO} = \\text{Recon} + \\beta \\cdot \\text{KL}\\).',
                    solution: 'Total loss \\(= 100 + 4 \\times 10 = 140\\). If we increase \\(\\beta\\) further, the optimizer will push harder to reduce KL, forcing the encoder to produce distributions closer to \\(\\mathcal{N}(0, I)\\). This means: (1) the latent space becomes more regular and potentially more disentangled, (2) the latent code carries less information about the specific input, (3) reconstruction quality degrades (blurrier outputs), and (4) in the extreme, posterior collapse occurs where \\(q_\\phi(\\mathbf{z} \\mid \\mathbf{x}) \\approx \\mathcal{N}(0, I)\\) for all \\(\\mathbf{x}\\), and the decoder ignores \\(\\mathbf{z}\\) entirely, producing the same average output for all inputs.'
                },
                {
                    question: 'A common complaint about VAEs is that they produce blurry images. Explain why this happens from the perspective of the Gaussian decoder assumption \\(p_\\theta(\\mathbf{x} \\mid \\mathbf{z}) = \\mathcal{N}(\\boldsymbol{\\mu}_\\theta(\\mathbf{z}), \\sigma^2 \\mathbf{I})\\) and MSE reconstruction loss.',
                    hint: 'What is the optimal \\(\\boldsymbol{\\mu}_\\theta(\\mathbf{z})\\) when \\(\\mathbf{z}\\) could correspond to multiple plausible images?',
                    solution: 'For a given \\(\\mathbf{z}\\), there may be multiple plausible images \\(\\mathbf{x}\\) that could have produced it (due to stochastic encoding and limited latent capacity). Under MSE loss (equivalently, Gaussian decoder), the optimal decoder output \\(\\boldsymbol{\\mu}_\\theta(\\mathbf{z})\\) is the <em>conditional mean</em> \\(\\mathbb{E}[\\mathbf{x} \\mid \\mathbf{z}]\\). If several sharp images are plausible for a given \\(\\mathbf{z}\\), the conditional mean is their <em>average</em>, which is blurry. For example, if \\(\\mathbf{z}\\) maps to either a "3" or an "8" with equal probability, the mean is a blurry superposition. Solutions include: using more expressive decoders (PixelCNN, autoregressive), adversarial losses (VAE-GAN), or perceptual losses that penalize structural dissimilarity rather than pixel-level MSE.'
                }
            ]
        }
    ]
});
