window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch21',
    number: 21,
    title: 'Vision & Multimodal Frontiers',
    subtitle: 'ViT, CLIP, Graph Neural Networks, Mixture of Experts, and State Space Models',
    sections: [
        // ============================================================
        // Section 1: Vision Transformer (ViT)
        // ============================================================
        {
            id: 'vision-transformer',
            title: 'Vision Transformer (ViT)',
            content: `
                <h2>Vision Transformer (ViT)</h2>

                <div class="env-block intuition">
                    <div class="env-title">From Convolutions to Tokens</div>
                    <div class="env-body"><p>For decades, convolutional neural networks dominated computer vision. They exploit local spatial structure by sliding learned filters across an image. But the Transformer, originally designed for language, processes <em>sequences of tokens</em>. The Vision Transformer (ViT), introduced by Dosovitskiy et al. (2020), asks a deceptively simple question: what if we treat an image as a sequence of patches and feed it directly into a standard Transformer encoder?</p></div>
                </div>

                <h3>Patch Embedding</h3>

                <p>Given an image \\(\\mathbf{x} \\in \\mathbb{R}^{H \\times W \\times C}\\), ViT divides it into a grid of non-overlapping patches, each of size \\(P \\times P\\). This produces \\(N = HW / P^2\\) patches. Each patch \\(\\mathbf{x}_p^{(i)} \\in \\mathbb{R}^{P^2 C}\\) is flattened into a vector and linearly projected into a \\(D\\)-dimensional embedding space:</p>

                \\[\\mathbf{z}_0^{(i)} = \\mathbf{x}_p^{(i)} \\mathbf{E} + \\mathbf{e}_{\\text{pos}}^{(i)}, \\quad \\mathbf{E} \\in \\mathbb{R}^{(P^2 C) \\times D}\\]

                <p>where \\(\\mathbf{e}_{\\text{pos}}^{(i)}\\) is a learnable position embedding that encodes the spatial location of the \\(i\\)-th patch. Without position embeddings, the Transformer would be permutation-invariant and lose all spatial information.</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (Patch Embedding)</div>
                    <div class="env-body"><p>The <strong>patch embedding</strong> maps each \\(P \\times P\\) image patch to a \\(D\\)-dimensional token via a learned linear projection \\(\\mathbf{E}\\). This is mathematically equivalent to a 2D convolution with kernel size \\(P\\) and stride \\(P\\).</p></div>
                </div>

                <h3>The CLS Token</h3>

                <p>ViT prepends a special learnable token \\(\\mathbf{z}_0^{(0)} = \\mathbf{x}_{\\text{class}}\\) to the sequence of patch embeddings. After passing through \\(L\\) Transformer encoder layers, the output corresponding to this CLS token, \\(\\mathbf{z}_L^{(0)}\\), serves as the global image representation for classification:</p>

                \\[\\hat{y} = \\text{MLP}(\\text{LayerNorm}(\\mathbf{z}_L^{(0)}))\\]

                <div class="env-block remark">
                    <div class="env-title">Why a CLS Token?</div>
                    <div class="env-body"><p>The CLS token acts as a "global aggregator." Through self-attention, it can attend to every patch token, accumulating information from the entire image. This mirrors BERT's use of a [CLS] token for sentence-level classification. An alternative is to simply average-pool all patch token outputs; in practice both approaches yield similar performance.</p></div>
                </div>

                <h3>The Full ViT Pipeline</h3>

                <p>The complete sequence passed to the Transformer encoder is:</p>

                \\[\\mathbf{z}_0 = [\\mathbf{x}_{\\text{class}};\\; \\mathbf{x}_p^{(1)}\\mathbf{E};\\; \\mathbf{x}_p^{(2)}\\mathbf{E};\\; \\dots;\\; \\mathbf{x}_p^{(N)}\\mathbf{E}] + \\mathbf{E}_{\\text{pos}}\\]

                <p>Each Transformer layer applies multi-head self-attention (MSA) and a feed-forward MLP with residual connections:</p>

                \\[\\mathbf{z}'_\\ell = \\text{MSA}(\\text{LN}(\\mathbf{z}_{\\ell-1})) + \\mathbf{z}_{\\ell-1}\\]
                \\[\\mathbf{z}_\\ell = \\text{MLP}(\\text{LN}(\\mathbf{z}'_\\ell)) + \\mathbf{z}'_\\ell\\]

                <div class="env-block example">
                    <div class="env-title">Example (ViT-Base Configuration)</div>
                    <div class="env-body">
                        <p>ViT-Base uses \\(P = 16\\), \\(D = 768\\), 12 layers, and 12 attention heads. For a \\(224 \\times 224\\) image, this gives \\(N = 196\\) patches. The input sequence length to the Transformer is 197 (196 patches + 1 CLS token). Total parameters: approximately 86M.</p>
                    </div>
                </div>

                <div class="env-block intuition">
                    <div class="env-title">Inductive Bias Trade-off</div>
                    <div class="env-body"><p>CNNs have strong inductive biases: locality (each filter sees a small neighborhood) and translation equivariance (the same filter applies everywhere). ViT lacks these biases. Self-attention is global from layer 1, and the model must <em>learn</em> locality from data. This means ViT underperforms CNNs on small datasets but excels when pretrained on large datasets (JFT-300M, ImageNet-21k) because it can learn more flexible representations unconstrained by fixed local receptive fields.</p></div>
                </div>

                <div class="viz-placeholder" data-viz="viz-vit-patches"></div>

                <h3>Position Embeddings: Learned vs. Sinusoidal</h3>

                <p>The original ViT uses 1D learned position embeddings, treating the 2D patch grid as a flattened sequence. Experiments show that 2D-aware position embeddings provide only marginal improvements, suggesting the model can recover 2D structure from learned 1D embeddings. The cosine similarity between position embeddings of nearby patches is high, confirming the model learns spatial proximity.</p>

                <div class="env-block warning">
                    <div class="env-title">Resolution Mismatch</div>
                    <div class="env-body"><p>If the input resolution at inference differs from training, the number of patches changes, so the learned position embeddings no longer align. The standard remedy is to 2D-interpolate the position embeddings to the new grid size. This is why ViT can fine-tune at higher resolutions than it was pretrained on, but the interpolation introduces a small performance cost.</p></div>
                </div>
            `,
            visualizations: [
                {
                    id: 'viz-vit-patches',
                    title: 'ViT Patch Embedding Pipeline',
                    description: 'Watch an image get split into patches, each patch flattened into a vector, then linearly projected into token embeddings. Adjust the patch size to see how the number of tokens changes.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 780, height: 420, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        var patchSize = 4; // patches per side: image divided into patchSize x patchSize grid
                        var animPhase = 0; // 0=image, 1=splitting, 2=flattening, 3=projecting
                        var animTime = 0;
                        var speed = 1;
                        var imgSize = 140;

                        VizEngine.createSlider(controls, 'Patches/side', 2, 8, patchSize, 1, function(v) {
                            patchSize = Math.round(v);
                            animPhase = 0;
                            animTime = 0;
                        });

                        VizEngine.createButton(controls, 'Restart', function() {
                            animPhase = 0;
                            animTime = 0;
                        });

                        // Generate a synthetic "image" with colored blocks
                        function getBlockColor(r, c, gridN) {
                            var hue = ((r * gridN + c) * 137.5) % 360;
                            return 'hsl(' + hue + ', 60%, 55%)';
                        }

                        function draw(t) {
                            animTime += 0.016 * speed;
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var N = patchSize;
                            var totalPatches = N * N;
                            var pSz = Math.floor(imgSize / N);
                            var imgX = 30, imgY = (H - imgSize) / 2;

                            // Phase transitions
                            if (animPhase === 0 && animTime > 1.5) { animPhase = 1; animTime = 0; }
                            if (animPhase === 1 && animTime > 2.0) { animPhase = 2; animTime = 0; }
                            if (animPhase === 2 && animTime > 2.0) { animPhase = 3; animTime = 0; }
                            if (animPhase === 3 && animTime > 3.0) { animPhase = 0; animTime = 0; }

                            var progress = Math.min(animTime / 1.5, 1);
                            var ease = progress * progress * (3 - 2 * progress); // smoothstep

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            var phases = ['Original Image', 'Splitting into ' + N + 'x' + N + ' patches', 'Flattening patches into vectors', 'Linear projection to D-dim tokens'];
                            ctx.fillText(phases[animPhase], W / 2, 8);

                            // Draw image / patches
                            for (var r = 0; r < N; r++) {
                                for (var c = 0; c < N; c++) {
                                    var col = getBlockColor(r, c, N);
                                    var baseX = imgX + c * pSz;
                                    var baseY = imgY + r * pSz;

                                    if (animPhase === 0) {
                                        // Solid image
                                        ctx.fillStyle = col;
                                        ctx.fillRect(baseX, baseY, pSz, pSz);
                                    } else if (animPhase === 1) {
                                        // Splitting: patches move apart
                                        var gap = ease * 4;
                                        var dx = (c - (N - 1) / 2) * gap;
                                        var dy = (r - (N - 1) / 2) * gap;
                                        ctx.fillStyle = col;
                                        ctx.fillRect(baseX + dx, baseY + dy, pSz - 1, pSz - 1);
                                        ctx.strokeStyle = viz.colors.white + '66';
                                        ctx.lineWidth = 1;
                                        ctx.strokeRect(baseX + dx, baseY + dy, pSz - 1, pSz - 1);
                                    } else if (animPhase === 2) {
                                        // Flattening: patches move to a column on the right
                                        var idx = r * N + c;
                                        var targetX = 220;
                                        var targetY = 30 + idx * (Math.min((H - 60) / totalPatches, 18));
                                        var flatW = Math.min(60, pSz * pSz / 2);
                                        var flatH = Math.min(14, (H - 60) / totalPatches - 2);

                                        var curX = baseX + (targetX - baseX) * ease;
                                        var curY = baseY + (targetY - baseY) * ease;
                                        var curW = pSz + (flatW - pSz) * ease;
                                        var curH = pSz + (flatH - pSz) * ease;

                                        ctx.fillStyle = col;
                                        ctx.fillRect(curX, curY, curW, curH);
                                    } else if (animPhase === 3) {
                                        // Projecting: flat vectors transform into embeddings
                                        var idx2 = r * N + c;
                                        var srcX = 220;
                                        var rowH = Math.min((H - 60) / totalPatches, 18);
                                        var srcY = 30 + idx2 * rowH;
                                        var flatW2 = Math.min(60, pSz * pSz / 2);
                                        var flatH2 = Math.min(14, rowH - 2);

                                        // Source flat vector
                                        ctx.fillStyle = col + '88';
                                        ctx.fillRect(srcX, srcY, flatW2, flatH2);

                                        // Arrow
                                        var arrowStartX = srcX + flatW2 + 10;
                                        var arrowEndX = 420;
                                        var arrowY = srcY + flatH2 / 2;
                                        ctx.strokeStyle = viz.colors.text + '55';
                                        ctx.lineWidth = 1;
                                        ctx.beginPath();
                                        ctx.moveTo(arrowStartX, arrowY);
                                        ctx.lineTo(arrowStartX + (arrowEndX - arrowStartX) * ease, arrowY);
                                        ctx.stroke();

                                        // Projection matrix E (just a label)
                                        if (idx2 === 0) {
                                            var matX = (arrowStartX + arrowEndX) / 2;
                                            ctx.fillStyle = viz.colors.purple;
                                            ctx.font = 'bold 12px -apple-system,sans-serif';
                                            ctx.textAlign = 'center';
                                            ctx.textBaseline = 'bottom';
                                            ctx.fillText('x E', matX, 24);
                                        }

                                        // Target: projected token
                                        var tokX = 430;
                                        var tokW = 120;
                                        ctx.globalAlpha = ease;
                                        ctx.fillStyle = col;
                                        ctx.fillRect(tokX, srcY, tokW * ease, flatH2);
                                        // Position embedding overlay
                                        ctx.fillStyle = viz.colors.yellow + '44';
                                        ctx.fillRect(tokX, srcY, tokW * ease, flatH2);
                                        ctx.globalAlpha = 1;

                                        // Token label
                                        if (ease > 0.7) {
                                            ctx.fillStyle = viz.colors.white;
                                            ctx.font = '10px -apple-system,sans-serif';
                                            ctx.textAlign = 'left';
                                            ctx.textBaseline = 'middle';
                                            ctx.fillText('z' + idx2, tokX + tokW * ease + 5, srcY + flatH2 / 2);
                                        }
                                    }
                                }
                            }

                            // CLS token indicator in phase 3
                            if (animPhase === 3 && ease > 0.3) {
                                ctx.globalAlpha = ease;
                                var clsY = 30 - Math.min((H - 60) / totalPatches, 18);
                                var tokX2 = 430;
                                var tokW2 = 120;
                                var clsH = Math.min(14, (H - 60) / totalPatches - 2);
                                ctx.fillStyle = viz.colors.red;
                                ctx.fillRect(tokX2, clsY > 10 ? clsY : 10, tokW2 * ease, clsH > 3 ? clsH : 10);
                                ctx.fillStyle = viz.colors.white;
                                ctx.font = 'bold 10px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('[CLS]', tokX2 + tokW2 * ease + 5, (clsY > 10 ? clsY : 10) + (clsH > 3 ? clsH : 10) / 2);
                                ctx.globalAlpha = 1;
                            }

                            // Labels
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            if (animPhase <= 1) {
                                ctx.fillText(N + 'x' + N + ' = ' + totalPatches + ' patches', imgX + imgSize / 2, imgY + imgSize + 10);
                            }
                            if (animPhase >= 2) {
                                ctx.fillText('Flat vectors', 250, H - 18);
                            }
                            if (animPhase >= 3) {
                                ctx.fillText('D-dim tokens + pos. emb.', 490, H - 18);
                            }

                            // Legend box
                            ctx.fillStyle = viz.colors.bg + 'cc';
                            ctx.fillRect(W - 200, H - 55, 190, 48);
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 1;
                            ctx.strokeRect(W - 200, H - 55, 190, 48);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Each color = one image patch', W - 190, H - 40);
                            ctx.fillStyle = viz.colors.yellow + '88';
                            ctx.fillRect(W - 190, H - 26, 12, 8);
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('= position embedding', W - 174, H - 22);
                        }

                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'An image of size \\(384 \\times 384 \\times 3\\) is fed to ViT with patch size \\(P=16\\). How many patch tokens are created? What is the dimensionality of each flattened patch before projection?',
                    hint: 'Number of patches = \\((H/P) \\times (W/P)\\). Each patch has \\(P^2 \\times C\\) values.',
                    solution: 'Number of patches: \\((384/16)^2 = 24^2 = 576\\). Each flattened patch has \\(16^2 \\times 3 = 768\\) dimensions. Including the CLS token, the Transformer receives a sequence of length 577.'
                },
                {
                    question: 'Why does ViT generally underperform CNNs when trained on small datasets like ImageNet-1k alone, yet outperform them on larger datasets?',
                    hint: 'Think about inductive biases. CNNs have built-in locality and translation equivariance; ViT must learn these from data.',
                    solution: 'CNNs encode locality and translation equivariance directly in their architecture, giving them strong priors that help on small datasets. ViT has no such inductive biases: every layer has global receptive fields and the model must learn spatial structure entirely from data. On small datasets, this flexibility becomes a liability (overfitting), but on very large datasets, it becomes an asset because ViT can discover representations that are more expressive than what CNNs can represent with their fixed local filters.'
                },
                {
                    question: 'If ViT is pretrained at resolution \\(224 \\times 224\\) with \\(P = 16\\), the model has \\(196 + 1 = 197\\) position embeddings. Explain what happens and what adjustment is needed if we fine-tune at resolution \\(448 \\times 448\\).',
                    hint: 'At \\(448 \\times 448\\), the number of patches becomes \\((448/16)^2 = 784\\). The model only has 196 position embeddings.',
                    solution: 'At \\(448 \\times 448\\) with \\(P = 16\\), we get \\(28 \\times 28 = 784\\) patches, but only 196 position embeddings exist. The solution is to reshape the \\(14 \\times 14\\) grid of learned position embeddings into a 2D map and bilinearly interpolate them to a \\(28 \\times 28\\) grid. This preserves approximate spatial relationships but is an approximation; a short fine-tuning phase at the new resolution is recommended.'
                }
            ]
        },

        // ============================================================
        // Section 2: CLIP & Contrastive Learning
        // ============================================================
        {
            id: 'clip-contrastive',
            title: 'CLIP & Contrastive Learning',
            content: `
                <h2>CLIP &amp; Contrastive Learning</h2>

                <div class="env-block intuition">
                    <div class="env-title">Learning from Natural Language Supervision</div>
                    <div class="env-body"><p>Traditional image classifiers learn a fixed set of categories: "cat," "dog," "car." But human understanding of images is far richer. CLIP (Contrastive Language-Image Pre-training, Radford et al. 2021) learns to associate images with free-form text descriptions. By training on 400 million image-text pairs scraped from the internet, CLIP learns a shared embedding space where images and their textual descriptions are pulled close together while non-matching pairs are pushed apart.</p></div>
                </div>

                <h3>Dual-Encoder Architecture</h3>

                <p>CLIP consists of two parallel encoders:</p>
                <ul>
                    <li><strong>Image encoder</strong>: either a ResNet or a ViT that maps an image \\(\\mathbf{x}\\) to a normalized vector \\(\\mathbf{f}(\\mathbf{x}) \\in \\mathbb{R}^d\\).</li>
                    <li><strong>Text encoder</strong>: a Transformer that maps a text string \\(t\\) to a normalized vector \\(\\mathbf{g}(t) \\in \\mathbb{R}^d\\).</li>
                </ul>

                <p>Both encoders project into the same \\(d\\)-dimensional space. The similarity between an image-text pair is measured by their cosine similarity:</p>

                \\[\\text{sim}(\\mathbf{x}, t) = \\mathbf{f}(\\mathbf{x})^\\top \\mathbf{g}(t)\\]

                <p>since both vectors are \\(\\ell_2\\)-normalized.</p>

                <h3>Contrastive Loss (InfoNCE)</h3>

                <div class="env-block definition">
                    <div class="env-title">Definition (Symmetric Contrastive Loss)</div>
                    <div class="env-body">
                        <p>Given a mini-batch of \\(N\\) image-text pairs \\(\\{(\\mathbf{x}_i, t_i)\\}_{i=1}^N\\), define the logits matrix \\(L_{ij} = \\tau \\cdot \\mathbf{f}(\\mathbf{x}_i)^\\top \\mathbf{g}(t_j)\\), where \\(\\tau\\) is a learnable temperature parameter. The loss is:</p>
                        <p>\\[\\mathcal{L} = \\frac{1}{2N} \\sum_{i=1}^{N} \\Big[ -\\log \\frac{\\exp(L_{ii})}{\\sum_j \\exp(L_{ij})} - \\log \\frac{\\exp(L_{ii})}{\\sum_j \\exp(L_{ji})} \\Big]\\]</p>
                        <p>The first term is "image-to-text" (which text best matches image \\(i\\)?), and the second is "text-to-image" (which image best matches text \\(i\\)?).</p>
                    </div>
                </div>

                <p>The diagonal entries \\(L_{ii}\\) correspond to the correct (positive) pairs. All off-diagonal entries \\(L_{ij}\\) for \\(i \\neq j\\) are negatives. The loss pushes positive pairs to have high similarity and negatives to have low similarity.</p>

                <div class="env-block remark">
                    <div class="env-title">Role of the Temperature \\(\\tau\\)</div>
                    <div class="env-body"><p>The temperature \\(\\tau\\) controls the "sharpness" of the softmax distribution over similarities. A higher \\(\\tau\\) makes the distribution more uniform (softer), while a lower \\(\\tau\\) makes it peakier, increasing the penalty for hard negatives. CLIP learns \\(\\tau\\) as a log-parameterized scalar, initialized to \\(\\tau = 1/0.07 \\approx 14.3\\).</p></div>
                </div>

                <div class="viz-placeholder" data-viz="viz-clip-matrix"></div>

                <h3>Zero-Shot Classification</h3>

                <p>After pretraining, CLIP can classify images into <em>any</em> set of categories without additional training. To classify an image into one of \\(K\\) classes:</p>

                <ol>
                    <li>Create text prompts for each class, e.g., "a photo of a {class name}."</li>
                    <li>Encode each prompt with the text encoder: \\(\\mathbf{g}(t_k)\\) for \\(k = 1, \\dots, K\\).</li>
                    <li>Encode the image: \\(\\mathbf{f}(\\mathbf{x})\\).</li>
                    <li>Predict the class with highest cosine similarity: \\(\\hat{k} = \\arg\\max_k \\mathbf{f}(\\mathbf{x})^\\top \\mathbf{g}(t_k)\\).</li>
                </ol>

                <div class="env-block example">
                    <div class="env-title">Example (Zero-Shot on ImageNet)</div>
                    <div class="env-body">
                        <p>CLIP achieves 76.2% top-1 accuracy on ImageNet zero-shot, matching a fully supervised ResNet-50 trained on 1.28M labeled images. Remarkably, CLIP was never trained on ImageNet labels; its supervision came entirely from image-text pairs found on the web.</p>
                    </div>
                </div>

                <div class="env-block intuition">
                    <div class="env-title">Why Contrastive Learning Works</div>
                    <div class="env-body"><p>Contrastive learning encodes a fundamental principle: <em>mutual information maximization</em>. By pulling matching image-text pairs together and pushing non-matching pairs apart, the model learns to capture the semantic content shared between the two modalities. The resulting embedding space is structured so that semantic similarity corresponds to geometric proximity. This is why CLIP can generalize to unseen categories: as long as the text describes a meaningful concept, the model can locate it in embedding space.</p></div>
                </div>

                <h3>Prompt Engineering and Ensembling</h3>

                <p>The choice of text prompt matters. "A photo of a dog" works better than just "dog" because the training data consists of natural language descriptions, not single-word labels. OpenAI found that using an ensemble of 80 prompt templates (e.g., "a photo of a {class}, a type of pet") and averaging the resulting text embeddings improved accuracy by 3-5% on ImageNet.</p>

                <div class="env-block warning">
                    <div class="env-title">Limitations of CLIP</div>
                    <div class="env-body"><p>CLIP struggles with fine-grained tasks (distinguishing car models or bird species), compositional reasoning ("a red cube on top of a blue sphere"), and counting. These weaknesses arise because contrastive learning optimizes for bag-of-concepts matching rather than structured scene understanding. Later models like SigLIP (sigmoid loss instead of softmax) and BLIP-2 address some of these limitations.</p></div>
                </div>
            `,
            visualizations: [
                {
                    id: 'viz-clip-matrix',
                    title: 'CLIP Contrastive Similarity Matrix',
                    description: 'Rows = images, columns = text descriptions. The diagonal (green) shows correct pairs with high similarity. Off-diagonal cells are negatives that should have low similarity. Adjust the temperature to see how it sharpens the distribution.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 780, height: 450, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        var temperature = 10.0;
                        var N = 6;

                        VizEngine.createSlider(controls, 'Temperature', 1, 30, temperature, 1, function(v) {
                            temperature = v;
                        });

                        // Simulated image-text pairs
                        var labels = ['cat', 'sunset', 'car', 'mountain', 'coffee', 'guitar'];
                        var imageLabels = labels.map(function(l) { return l; });
                        var textLabels = labels.map(function(l) { return '"a photo of a ' + l + '"'; });

                        // Simulated embedding similarities (raw cosine, before temperature)
                        // Diagonal should be high, off-diagonal low
                        var rawSim = [];
                        for (var i = 0; i < N; i++) {
                            rawSim[i] = [];
                            for (var j = 0; j < N; j++) {
                                if (i === j) {
                                    rawSim[i][j] = 0.28 + Math.random() * 0.07;
                                } else {
                                    // Some pairs more confusable
                                    rawSim[i][j] = -0.05 + Math.random() * 0.15;
                                }
                            }
                        }

                        function softmax(arr) {
                            var maxVal = -Infinity;
                            for (var k = 0; k < arr.length; k++) { if (arr[k] > maxVal) maxVal = arr[k]; }
                            var sumExp = 0;
                            var result = [];
                            for (var k = 0; k < arr.length; k++) {
                                result[k] = Math.exp(arr[k] - maxVal);
                                sumExp += result[k];
                            }
                            for (var k = 0; k < arr.length; k++) { result[k] /= sumExp; }
                            return result;
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var cellSize = 48;
                            var matrixLeft = 180;
                            var matrixTop = 70;

                            // Compute scaled logits
                            var logits = [];
                            for (var i = 0; i < N; i++) {
                                logits[i] = [];
                                for (var j = 0; j < N; j++) {
                                    logits[i][j] = rawSim[i][j] * temperature;
                                }
                            }

                            // Compute softmax probabilities per row (image-to-text)
                            var probsI2T = [];
                            for (var i = 0; i < N; i++) {
                                probsI2T[i] = softmax(logits[i]);
                            }

                            // Draw column headers (text labels)
                            ctx.fillStyle = viz.colors.teal;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'bottom';
                            for (var j = 0; j < N; j++) {
                                ctx.save();
                                ctx.translate(matrixLeft + j * cellSize + cellSize / 2, matrixTop - 8);
                                ctx.rotate(-0.5);
                                ctx.fillText(textLabels[j], 0, 0);
                                ctx.restore();
                            }

                            // Draw row headers (image labels)
                            ctx.fillStyle = viz.colors.orange;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'middle';
                            for (var i = 0; i < N; i++) {
                                ctx.fillText(imageLabels[i], matrixLeft - 12, matrixTop + i * cellSize + cellSize / 2);
                                // small image icon
                                var iconX = matrixLeft - 70;
                                var iconY = matrixTop + i * cellSize + cellSize / 2 - 10;
                                ctx.strokeStyle = viz.colors.orange + '88';
                                ctx.lineWidth = 1;
                                ctx.strokeRect(iconX, iconY, 18, 18);
                                ctx.fillStyle = viz.colors.orange + '33';
                                ctx.fillRect(iconX, iconY, 18, 18);
                            }

                            // Draw matrix cells
                            for (var i = 0; i < N; i++) {
                                for (var j = 0; j < N; j++) {
                                    var x = matrixLeft + j * cellSize;
                                    var y = matrixTop + i * cellSize;
                                    var prob = probsI2T[i][j];

                                    // Color: diagonal = green, off-diagonal = blue/red
                                    var r2, g2, b2;
                                    if (i === j) {
                                        r2 = Math.round(30 + prob * 30);
                                        g2 = Math.round(100 + prob * 155);
                                        b2 = Math.round(50 + prob * 30);
                                    } else {
                                        r2 = Math.round(40 + prob * 200);
                                        g2 = Math.round(30 + prob * 30);
                                        b2 = Math.round(60 + prob * 60);
                                    }
                                    ctx.fillStyle = 'rgb(' + r2 + ',' + g2 + ',' + b2 + ')';
                                    ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);

                                    // Diagonal highlight border
                                    if (i === j) {
                                        ctx.strokeStyle = viz.colors.green;
                                        ctx.lineWidth = 2;
                                        ctx.strokeRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
                                    }

                                    // Probability text
                                    ctx.fillStyle = prob > 0.5 ? '#000' : viz.colors.white;
                                    ctx.font = '11px monospace';
                                    ctx.textAlign = 'center';
                                    ctx.textBaseline = 'middle';
                                    ctx.fillText(prob.toFixed(2), x + cellSize / 2, y + cellSize / 2);
                                }
                            }

                            // Title & axis labels
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Contrastive Similarity Matrix (softmax across each row)', W / 2, 8);

                            ctx.fillStyle = viz.colors.orange;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Images', matrixLeft - 12, matrixTop - 20);

                            ctx.fillStyle = viz.colors.teal;
                            ctx.textAlign = 'left';
                            ctx.fillText('Texts', matrixLeft + N * cellSize + 10, matrixTop - 20);

                            // Loss explanation on the right
                            var infoX = matrixLeft + N * cellSize + 30;
                            var infoY = matrixTop + 20;
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Contrastive goal:', infoX, infoY);
                            ctx.fillStyle = viz.colors.green;
                            ctx.fillText('Diagonal = 1.0', infoX, infoY + 20);
                            ctx.fillStyle = viz.colors.red;
                            ctx.fillText('Off-diagonal = 0.0', infoX, infoY + 40);

                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillText('Higher temperature', infoX, infoY + 75);
                            ctx.fillText('= sharper softmax', infoX, infoY + 90);
                            ctx.fillText('= stronger contrast', infoX, infoY + 105);

                            // Compute and display loss
                            var loss = 0;
                            for (var i = 0; i < N; i++) {
                                loss -= Math.log(probsI2T[i][i] + 1e-10);
                            }
                            loss /= N;

                            ctx.fillStyle = viz.colors.yellow;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.fillText('Loss = ' + loss.toFixed(3), infoX, infoY + 140);

                            // Legend
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textBaseline = 'bottom';
                            ctx.textAlign = 'center';
                            ctx.fillText('Green border = positive pair (matched image-text). Values show softmax probability.', W / 2, H - 8);
                        }

                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'In a CLIP training batch of size \\(N = 512\\), how many positive pairs and how many negative pairs are there? What is the ratio?',
                    hint: 'Each of the 512 image-text pairs is a positive. Each image is compared to all 512 texts, so there are \\(N-1\\) negatives per image.',
                    solution: 'Positive pairs: \\(N = 512\\). For each image, there are \\(N - 1 = 511\\) negatives, giving \\(N(N-1) = 512 \\times 511 = 261{,}632\\) negative pairings in the image-to-text direction. The ratio of negatives to positives per sample is \\(511:1\\). Symmetrically, the text-to-image direction has the same count. In total across both directions: \\(512\\) positive logits and \\(2 \\times 512 \\times 511 = 523{,}264\\) negative logits.'
                },
                {
                    question: 'Explain why CLIP can perform zero-shot classification on a completely new dataset it has never seen, while a standard ResNet-50 trained on ImageNet cannot.',
                    hint: 'Think about what each model learns: fixed categories vs. a shared image-text embedding space.',
                    solution: 'A standard ResNet-50 has a fixed classification head with a weight vector for each of the 1000 ImageNet classes. To classify a new category, one must add new weights and fine-tune with labeled data. CLIP, by contrast, learns a shared embedding space for images and text. Any textual description can be encoded into this space, so new categories are introduced simply by writing a text prompt. No retraining or labeled data is needed; the model merely checks which text embedding is closest to the image embedding.'
                },
                {
                    question: 'If we set the temperature \\(\\tau\\) extremely high (say \\(\\tau \\to \\infty\\)), what happens to the contrastive loss? What about \\(\\tau \\to 0\\)?',
                    hint: 'Consider the effect on the softmax distribution. As \\(\\tau \\to \\infty\\), logits become huge; as \\(\\tau \\to 0\\), logits vanish.',
                    solution: 'As \\(\\tau \\to \\infty\\), all logits \\(\\tau \\cdot \\text{sim}_{ij}\\) become very large. However, the softmax depends on <em>differences</em> between logits, which scale linearly with \\(\\tau\\). If the positive pair has even slightly higher raw similarity, the softmax probability concentrates entirely on the positive, driving the loss to zero. In practice, excessively large \\(\\tau\\) can cause numerical overflow. As \\(\\tau \\to 0\\), all logits approach zero, making the softmax distribution uniform: \\(p_{ij} \\to 1/N\\) for all \\(j\\). The loss approaches \\(\\log N\\), which provides no useful gradient signal for distinguishing positives from negatives.'
                }
            ]
        },

        // ============================================================
        // Section 3: Graph Neural Networks
        // ============================================================
        {
            id: 'graph-neural-networks',
            title: 'Graph Neural Networks',
            content: `
                <h2>Graph Neural Networks</h2>

                <div class="env-block intuition">
                    <div class="env-title">Beyond Grids and Sequences</div>
                    <div class="env-body"><p>Images live on regular 2D grids; text is a 1D sequence. But many important data types are neither: social networks, molecular structures, knowledge graphs, citation networks, and protein interaction maps are all naturally represented as <em>graphs</em>. Graph Neural Networks (GNNs) generalize deep learning to these irregular domains by defining neural network operations on graph-structured data.</p></div>
                </div>

                <h3>Graph Basics</h3>

                <p>A graph \\(G = (V, E)\\) consists of a set of nodes \\(V = \\{v_1, \\dots, v_n\\}\\) and edges \\(E \\subseteq V \\times V\\). Each node \\(v_i\\) has a feature vector \\(\\mathbf{h}_i^{(0)} \\in \\mathbb{R}^d\\). The graph structure is encoded in the adjacency matrix \\(\\mathbf{A} \\in \\{0, 1\\}^{n \\times n}\\), where \\(A_{ij} = 1\\) if there is an edge from \\(v_i\\) to \\(v_j\\).</p>

                <h3>Message Passing Framework</h3>

                <div class="env-block definition">
                    <div class="env-title">Definition (Message Passing Neural Network)</div>
                    <div class="env-body">
                        <p>A <strong>message passing neural network</strong> (MPNN) updates each node's representation over \\(L\\) rounds. At round \\(\\ell\\), the update for node \\(v_i\\) consists of three steps:</p>
                        <ol>
                            <li><strong>Message</strong>: each neighbor \\(v_j \\in \\mathcal{N}(v_i)\\) sends a message \\(\\mathbf{m}_{j \\to i}^{(\\ell)} = \\text{MSG}^{(\\ell)}(\\mathbf{h}_i^{(\\ell-1)}, \\mathbf{h}_j^{(\\ell-1)}, \\mathbf{e}_{ji})\\)</li>
                            <li><strong>Aggregate</strong>: messages are combined via a permutation-invariant function \\(\\mathbf{m}_i^{(\\ell)} = \\text{AGG}(\\{\\mathbf{m}_{j \\to i}^{(\\ell)} : v_j \\in \\mathcal{N}(v_i)\\})\\)</li>
                            <li><strong>Update</strong>: \\(\\mathbf{h}_i^{(\\ell)} = \\text{UPDATE}^{(\\ell)}(\\mathbf{h}_i^{(\\ell-1)}, \\mathbf{m}_i^{(\\ell)})\\)</li>
                        </ol>
                    </div>
                </div>

                <p>The aggregation function must be <em>permutation-invariant</em> (e.g., sum, mean, max) because nodes have no canonical ordering. After \\(L\\) rounds, each node's representation \\(\\mathbf{h}_i^{(L)}\\) has incorporated information from its \\(L\\)-hop neighborhood.</p>

                <h3>Graph Convolutional Network (GCN)</h3>

                <p>The GCN (Kipf &amp; Welling, 2017) is a simple but effective instantiation of message passing. The layer-wise update rule is:</p>

                \\[\\mathbf{H}^{(\\ell)} = \\sigma\\Big(\\tilde{\\mathbf{D}}^{-1/2} \\tilde{\\mathbf{A}} \\tilde{\\mathbf{D}}^{-1/2} \\mathbf{H}^{(\\ell-1)} \\mathbf{W}^{(\\ell)}\\Big)\\]

                <p>where \\(\\tilde{\\mathbf{A}} = \\mathbf{A} + \\mathbf{I}\\) is the adjacency matrix with self-loops, \\(\\tilde{\\mathbf{D}}\\) is the corresponding degree matrix, and \\(\\mathbf{W}^{(\\ell)}\\) is a learnable weight matrix. The normalization \\(\\tilde{\\mathbf{D}}^{-1/2} \\tilde{\\mathbf{A}} \\tilde{\\mathbf{D}}^{-1/2}\\) computes a symmetric, degree-normalized average of neighbors' features.</p>

                <div class="env-block remark">
                    <div class="env-title">GCN as Spectral Filtering</div>
                    <div class="env-body"><p>The GCN propagation rule can be derived as a first-order approximation to spectral graph convolutions. The normalized adjacency \\(\\tilde{\\mathbf{D}}^{-1/2} \\tilde{\\mathbf{A}} \\tilde{\\mathbf{D}}^{-1/2}\\) acts as a low-pass filter on the graph: it smooths node features by averaging them with their neighbors. This is why GCN works well for tasks with homophily (connected nodes tend to have the same label) but can struggle with heterophily.</p></div>
                </div>

                <h3>Graph Attention Network (GAT)</h3>

                <p>GCN treats all neighbors equally (weighted only by degree). The Graph Attention Network (GAT; Velickovic et al., 2018) introduces <em>learned attention coefficients</em>:</p>

                \\[\\alpha_{ij} = \\frac{\\exp\\big(\\text{LeakyReLU}(\\mathbf{a}^\\top [\\mathbf{W}\\mathbf{h}_i \\| \\mathbf{W}\\mathbf{h}_j])\\big)}{\\sum_{k \\in \\mathcal{N}(i)} \\exp\\big(\\text{LeakyReLU}(\\mathbf{a}^\\top [\\mathbf{W}\\mathbf{h}_i \\| \\mathbf{W}\\mathbf{h}_k])\\big)}\\]

                <p>where \\(\\|\\) denotes concatenation and \\(\\mathbf{a}\\) is a learnable attention vector. The update becomes:</p>

                \\[\\mathbf{h}_i^{(\\ell)} = \\sigma\\Big(\\sum_{j \\in \\mathcal{N}(i)} \\alpha_{ij} \\mathbf{W} \\mathbf{h}_j^{(\\ell-1)}\\Big)\\]

                <p>Multi-head attention is used analogously to Transformers: \\(K\\) independent attention heads are computed and concatenated (or averaged in the final layer).</p>

                <div class="viz-placeholder" data-viz="viz-gnn-message"></div>

                <div class="env-block example">
                    <div class="env-title">Example (Molecular Property Prediction)</div>
                    <div class="env-body">
                        <p>In drug discovery, a molecule is represented as a graph where atoms are nodes and bonds are edges. Each atom carries features (element type, charge, hybridization). A GNN processes the molecular graph and outputs a graph-level prediction (e.g., toxicity or binding affinity) by first updating atom features via message passing, then pooling all atom features into a single graph-level vector.</p>
                    </div>
                </div>

                <div class="env-block warning">
                    <div class="env-title">Over-Smoothing</div>
                    <div class="env-body"><p>With too many message passing rounds, all node features converge to a similar vector because information diffuses across the entire graph. This is called <strong>over-smoothing</strong>. For most GNN architectures, 2-4 layers work best. Mitigation strategies include residual connections, jumping knowledge networks, and PairNorm.</p></div>
                </div>

                <h3>Expressivity and the WL Test</h3>

                <p>The Weisfeiler-Leman (WL) graph isomorphism test iteratively refines node labels by hashing each node's neighborhood. Xu et al. (2019) proved that message passing GNNs are <em>at most</em> as powerful as the 1-WL test in distinguishing graph structures. The Graph Isomorphism Network (GIN) achieves this upper bound by using sum aggregation and MLPs for the update function. More expressive architectures require higher-order message passing or subgraph counting.</p>
            `,
            visualizations: [
                {
                    id: 'viz-gnn-message',
                    title: 'GNN Message Passing Animation',
                    description: 'Watch nodes collect and aggregate messages from their neighbors over multiple rounds. Node colors represent feature vectors that evolve through message passing. Click "Step" to advance one round, or "Auto" to animate.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 780, height: 420, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var centerX = W / 2 - 60;
                        var centerY = H / 2;

                        var round = 0;
                        var maxRounds = 5;
                        var animProgress = 1.0; // 1 = done with current round
                        var autoMode = false;

                        // Graph structure: 8 nodes with edges
                        var nodes = [
                            { x: centerX - 120, y: centerY - 80 },
                            { x: centerX - 40, y: centerY - 130 },
                            { x: centerX + 80, y: centerY - 100 },
                            { x: centerX + 140, y: centerY - 20 },
                            { x: centerX + 80, y: centerY + 90 },
                            { x: centerX - 30, y: centerY + 110 },
                            { x: centerX - 130, y: centerY + 50 },
                            { x: centerX + 10, y: centerY }
                        ];

                        var edges = [
                            [0, 1], [0, 6], [0, 7],
                            [1, 2], [1, 7],
                            [2, 3], [2, 7],
                            [3, 4],
                            [4, 5], [4, 7],
                            [5, 6], [5, 7],
                            [6, 7]
                        ];

                        // Initial colors (hue)
                        var nodeHues = [0, 45, 90, 135, 180, 225, 270, 315];
                        var nodeFeatures = []; // [round][node] = hue
                        nodeFeatures[0] = nodeHues.slice();

                        // Neighbors
                        var neighbors = [];
                        for (var i = 0; i < nodes.length; i++) { neighbors[i] = []; }
                        for (var e = 0; e < edges.length; e++) {
                            var a = edges[e][0], b = edges[e][1];
                            neighbors[a].push(b);
                            neighbors[b].push(a);
                        }

                        function computeRound(r) {
                            if (nodeFeatures[r]) return;
                            nodeFeatures[r] = [];
                            for (var i = 0; i < nodes.length; i++) {
                                // Average of self + neighbors hues (simplified message passing)
                                var sum = nodeFeatures[r - 1][i];
                                var count = 1;
                                for (var k = 0; k < neighbors[i].length; k++) {
                                    sum += nodeFeatures[r - 1][neighbors[i][k]];
                                    count++;
                                }
                                nodeFeatures[r][i] = sum / count;
                            }
                        }

                        VizEngine.createButton(controls, 'Step', function() {
                            if (round < maxRounds && animProgress >= 1.0) {
                                round++;
                                computeRound(round);
                                animProgress = 0;
                                autoMode = false;
                            }
                        });

                        VizEngine.createButton(controls, 'Auto', function() {
                            autoMode = !autoMode;
                        });

                        VizEngine.createButton(controls, 'Reset', function() {
                            round = 0;
                            animProgress = 1.0;
                            autoMode = false;
                            nodeFeatures = [nodeHues.slice()];
                        });

                        function draw(t) {
                            // Auto mode
                            if (autoMode && animProgress >= 1.0 && round < maxRounds) {
                                round++;
                                computeRound(round);
                                animProgress = 0;
                            }

                            if (animProgress < 1.0) {
                                animProgress += 0.012;
                                if (animProgress > 1.0) animProgress = 1.0;
                            }

                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var ease = animProgress * animProgress * (3 - 2 * animProgress);

                            // Draw edges
                            for (var e = 0; e < edges.length; e++) {
                                var a = edges[e][0], b = edges[e][1];
                                ctx.strokeStyle = viz.colors.grid;
                                ctx.lineWidth = 1.5;
                                ctx.beginPath();
                                ctx.moveTo(nodes[a].x, nodes[a].y);
                                ctx.lineTo(nodes[b].x, nodes[b].y);
                                ctx.stroke();
                            }

                            // Draw message particles during animation
                            if (animProgress < 1.0 && round > 0) {
                                for (var e = 0; e < edges.length; e++) {
                                    var a = edges[e][0], b = edges[e][1];
                                    // Message from a to b
                                    var mx1 = nodes[a].x + (nodes[b].x - nodes[a].x) * ease;
                                    var my1 = nodes[a].y + (nodes[b].y - nodes[a].y) * ease;
                                    var hueA = nodeFeatures[round - 1][a];
                                    ctx.fillStyle = 'hsl(' + hueA + ', 70%, 60%)';
                                    ctx.globalAlpha = 0.7;
                                    ctx.beginPath();
                                    ctx.arc(mx1, my1, 4, 0, Math.PI * 2);
                                    ctx.fill();

                                    // Message from b to a
                                    var mx2 = nodes[b].x + (nodes[a].x - nodes[b].x) * ease;
                                    var my2 = nodes[b].y + (nodes[a].y - nodes[b].y) * ease;
                                    var hueB = nodeFeatures[round - 1][b];
                                    ctx.fillStyle = 'hsl(' + hueB + ', 70%, 60%)';
                                    ctx.beginPath();
                                    ctx.arc(mx2, my2, 4, 0, Math.PI * 2);
                                    ctx.fill();
                                    ctx.globalAlpha = 1.0;
                                }
                            }

                            // Draw nodes
                            var displayRound = (animProgress >= 1.0) ? round : round - 1;
                            if (displayRound < 0) displayRound = 0;
                            for (var i = 0; i < nodes.length; i++) {
                                var hue;
                                if (animProgress >= 1.0 || round === 0) {
                                    hue = nodeFeatures[displayRound][i];
                                } else {
                                    // Interpolate from old to new
                                    var oldHue = nodeFeatures[round - 1][i];
                                    var newHue = nodeFeatures[round][i];
                                    hue = oldHue + (newHue - oldHue) * ease;
                                }

                                // Node glow
                                ctx.fillStyle = 'hsl(' + hue + ', 70%, 60%)';
                                ctx.globalAlpha = 0.2;
                                ctx.beginPath();
                                ctx.arc(nodes[i].x, nodes[i].y, 26, 0, Math.PI * 2);
                                ctx.fill();
                                ctx.globalAlpha = 1.0;

                                // Node circle
                                ctx.fillStyle = 'hsl(' + hue + ', 70%, 50%)';
                                ctx.beginPath();
                                ctx.arc(nodes[i].x, nodes[i].y, 18, 0, Math.PI * 2);
                                ctx.fill();

                                ctx.strokeStyle = 'hsl(' + hue + ', 70%, 70%)';
                                ctx.lineWidth = 2;
                                ctx.beginPath();
                                ctx.arc(nodes[i].x, nodes[i].y, 18, 0, Math.PI * 2);
                                ctx.stroke();

                                // Node label
                                ctx.fillStyle = viz.colors.white;
                                ctx.font = 'bold 12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('v' + i, nodes[i].x, nodes[i].y);
                            }

                            // Info panel on the right
                            var panelX = W - 210;
                            ctx.fillStyle = viz.colors.bg + 'cc';
                            ctx.fillRect(panelX, 20, 200, 180);
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 1;
                            ctx.strokeRect(panelX, 20, 200, 180);

                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Message Passing', panelX + 12, 28);

                            ctx.fillStyle = viz.colors.teal;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.fillText('Round: ' + round + ' / ' + maxRounds, panelX + 12, 52);

                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillText('Each round:', panelX + 12, 78);
                            ctx.fillText('1. Neighbors send messages', panelX + 12, 96);
                            ctx.fillText('2. Aggregate (mean)', panelX + 12, 112);
                            ctx.fillText('3. Update node feature', panelX + 12, 128);

                            ctx.fillStyle = viz.colors.yellow;
                            ctx.fillText('After ' + round + ' rounds, each node', panelX + 12, 152);
                            ctx.fillText('sees its ' + round + '-hop neighborhood', panelX + 12, 168);

                            // Bottom text
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'bottom';
                            ctx.fillText('Node colors converge as features are aggregated (over-smoothing at high depth)', W / 2, H - 8);
                        }

                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'In a GCN with 3 layers, what is the receptive field of each node? That is, how far away in the graph can information propagate to reach a given node?',
                    hint: 'Each layer aggregates information from 1-hop neighbors. Multiple layers compose.',
                    solution: 'After \\(L\\) layers of GCN, each node has aggregated information from its \\(L\\)-hop neighborhood. With 3 layers, each node receives messages from all nodes within 3 hops (graph distance \\(\\leq 3\\)). If the graph has diameter \\(\\leq 3\\), every node sees every other node.'
                },
                {
                    question: 'Why must the aggregation function in a GNN be permutation-invariant? Give two examples of valid aggregation functions and one example that is not valid.',
                    hint: 'Nodes have no canonical ordering. If we relabel the neighbors, the output should not change.',
                    solution: 'Graphs have no canonical node ordering; the neighbors of a node form a <em>set</em>, not a sequence. If the aggregation depended on the order in which neighbors are processed, the output would change arbitrarily under relabeling, making the network inconsistent. Valid aggregation functions: <strong>sum</strong> (\\(\\sum_j \\mathbf{m}_j\\)), <strong>mean</strong> (\\(\\frac{1}{|\\mathcal{N}|} \\sum_j \\mathbf{m}_j\\)), and <strong>max</strong> (element-wise maximum). An invalid aggregation: concatenation in a fixed order, since permuting the neighbors would produce a different concatenated vector.'
                },
                {
                    question: 'Explain the over-smoothing problem in GNNs. Why does stacking more layers not always help, unlike in CNNs where deeper networks tend to be more powerful?',
                    hint: 'Think about what happens to node features as they are repeatedly averaged with their neighbors across many rounds.',
                    solution: 'In each message passing round, every node mixes its features with its neighbors. After many rounds, this repeated averaging acts like a diffusion process: node features converge to a common value (the graph\'s leading eigenvector of the normalized adjacency). All nodes become indistinguishable, destroying the discriminative information needed for downstream tasks. This is fundamentally different from CNNs, where the local receptive field grows gradually and residual connections preserve information. GNNs typically work best with 2-4 layers. Deeper GNNs require architectural innovations like residual connections, skip connections, or normalization strategies like PairNorm.'
                }
            ]
        },

        // ============================================================
        // Section 4: Mixture of Experts
        // ============================================================
        {
            id: 'mixture-of-experts',
            title: 'Mixture of Experts',
            content: `
                <h2>Mixture of Experts</h2>

                <div class="env-block intuition">
                    <div class="env-title">Scaling Without Proportional Compute</div>
                    <div class="env-body"><p>A central tension in deep learning: larger models have more capacity and tend to perform better, but they also require proportionally more computation for every input. The <strong>Mixture of Experts</strong> (MoE) architecture breaks this link. By routing each input to only a small subset of "expert" subnetworks, an MoE model can have vastly more parameters than a dense model while using similar compute per forward pass.</p></div>
                </div>

                <h3>Architecture</h3>

                <p>An MoE layer replaces the standard feed-forward network (FFN) in a Transformer block with \\(E\\) parallel expert networks \\(\\{\\text{FFN}_1, \\dots, \\text{FFN}_E\\}\\) and a gating (router) network. Given an input token \\(\\mathbf{x}\\):</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (Sparse Gating MoE)</div>
                    <div class="env-body">
                        <p>The <strong>router</strong> computes a score for each expert:</p>
                        <p>\\[\\mathbf{s} = \\text{softmax}(\\mathbf{W}_g \\mathbf{x})\\]</p>
                        <p>Only the top-\\(k\\) experts (typically \\(k = 1\\) or \\(k = 2\\)) are activated. Let \\(\\mathcal{T}_k\\) be the set of top-\\(k\\) expert indices. The output is:</p>
                        <p>\\[\\text{MoE}(\\mathbf{x}) = \\sum_{i \\in \\mathcal{T}_k} s_i \\cdot \\text{FFN}_i(\\mathbf{x})\\]</p>
                        <p>where \\(s_i\\) are the renormalized gating weights over the selected experts.</p>
                    </div>
                </div>

                <p>Because only \\(k \\ll E\\) experts process each token, the per-token computation is \\(k/E\\) of what a dense model with the same total parameters would require. For example, with \\(E = 64\\) experts and \\(k = 2\\), each token uses roughly 1/32 of the total expert parameters.</p>

                <div class="env-block example">
                    <div class="env-title">Example (Switch Transformer)</div>
                    <div class="env-body">
                        <p>The Switch Transformer (Fedus et al., 2022) uses \\(k = 1\\) (route each token to exactly one expert). With 128 experts, the model has 1.6 trillion parameters but activates only ~16 billion per input token, achieving 4-7x speedup over dense baselines with similar quality. Mixtral 8x7B uses 8 experts with top-2 routing, yielding 47B total parameters but using only about 13B per token.</p>
                    </div>
                </div>

                <h3>Load Balancing</h3>

                <p>A key challenge is <strong>load imbalance</strong>: the router may learn to route most tokens to a few popular experts while others sit idle. This wastes capacity and creates compute bottlenecks (in distributed systems, the busiest expert determines overall latency).</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (Auxiliary Load Balancing Loss)</div>
                    <div class="env-body">
                        <p>To encourage balanced routing, an auxiliary loss is added:</p>
                        <p>\\[\\mathcal{L}_{\\text{balance}} = \\alpha \\cdot E \\cdot \\sum_{i=1}^{E} f_i \\cdot p_i\\]</p>
                        <p>where \\(f_i\\) is the fraction of tokens routed to expert \\(i\\) and \\(p_i\\) is the average router probability assigned to expert \\(i\\). The product \\(f_i \\cdot p_i\\) is minimized when both are uniform (\\(1/E\\) each). The coefficient \\(\\alpha\\) (typically 0.01) controls the strength of this regularization.</p>
                    </div>
                </div>

                <div class="env-block remark">
                    <div class="env-title">Expert Specialization</div>
                    <div name="env-body"><div class="env-body"><p>Despite the load balancing loss, experts do develop specializations. Analysis of trained MoE models reveals that different experts specialize in different token types, syntactic structures, or semantic domains. For instance, one expert might handle punctuation and formatting tokens while another focuses on named entities. This emergent specialization is part of why MoE achieves better performance than simply averaging expert outputs.</p></div></div>
                </div>

                <div class="viz-placeholder" data-viz="viz-moe-routing"></div>

                <h3>Training Challenges</h3>

                <p>MoE models introduce several practical difficulties:</p>

                <ul>
                    <li><strong>Training instability</strong>: Router collapse (all tokens going to one expert) can occur early in training. Solutions include expert dropout, noise injection into router logits, and careful initialization.</li>
                    <li><strong>Communication overhead</strong>: In distributed training, tokens must be dispatched to the correct expert on the correct device, requiring all-to-all communication that can dominate training time.</li>
                    <li><strong>Memory</strong>: Although computation is sparse, all expert parameters must be stored. A 1T-parameter MoE model requires the same memory as a 1T dense model, even though inference FLOPS are much lower.</li>
                </ul>

                <div class="env-block warning">
                    <div class="env-title">MoE vs. Dense: When to Use Which</div>
                    <div class="env-body"><p>MoE is most beneficial when you can afford the memory for many parameters but want to keep per-token compute low (e.g., serving large language models at scale). For small-scale training or memory-constrained settings, dense models are simpler and more straightforward to train. The engineering complexity of MoE (routing, load balancing, distributed expert placement) is nontrivial.</p></div>
                </div>

                <h3>Capacity Factor and Token Dropping</h3>

                <p>In practice, each expert has a <em>capacity factor</em> \\(C\\) that limits the maximum number of tokens it can process. If the number of tokens routed to an expert exceeds its capacity, the overflow tokens are either dropped (processed by a residual connection only) or rerouted. The capacity factor is typically set to 1.0-1.25, balancing compute efficiency against token dropping rate.</p>
            `,
            visualizations: [
                {
                    id: 'viz-moe-routing',
                    title: 'Mixture of Experts: Token Routing',
                    description: 'Input tokens are routed to different experts by the gating network. Adjust top-k to see how many experts each token is sent to. Watch how gating weights determine the expert selection.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 780, height: 460, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        var topK = 2;
                        var numExperts = 6;
                        var numTokens = 8;
                        var animTime = 0;

                        VizEngine.createSlider(controls, 'Top-k', 1, 4, topK, 1, function(v) {
                            topK = Math.round(v);
                            regenerateRouting();
                        });

                        VizEngine.createButton(controls, 'New Routing', function() {
                            regenerateRouting();
                        });

                        // Generate random routing scores
                        var routerScores = [];
                        var routerProbs = [];
                        var selectedExperts = [];

                        function softmaxArr(arr) {
                            var maxV = -Infinity;
                            for (var i = 0; i < arr.length; i++) { if (arr[i] > maxV) maxV = arr[i]; }
                            var sumE = 0;
                            var result = [];
                            for (var i = 0; i < arr.length; i++) {
                                result[i] = Math.exp(arr[i] - maxV);
                                sumE += result[i];
                            }
                            for (var i = 0; i < arr.length; i++) { result[i] /= sumE; }
                            return result;
                        }

                        function regenerateRouting() {
                            routerScores = [];
                            routerProbs = [];
                            selectedExperts = [];
                            for (var t = 0; t < numTokens; t++) {
                                var scores = [];
                                for (var e = 0; e < numExperts; e++) {
                                    scores.push((Math.random() - 0.3) * 3);
                                }
                                routerScores.push(scores);
                                routerProbs.push(softmaxArr(scores));

                                // Select top-k
                                var indexed = scores.map(function(s, idx) { return { score: s, idx: idx }; });
                                indexed.sort(function(a, b) { return b.score - a.score; });
                                var sel = [];
                                for (var i = 0; i < Math.min(topK, numExperts); i++) {
                                    sel.push(indexed[i].idx);
                                }
                                selectedExperts.push(sel);
                            }
                            animTime = 0;
                        }

                        regenerateRouting();

                        var tokenLabels = ['the', 'cat', 'sat', 'on', 'the', 'warm', 'soft', 'mat'];
                        var expertColors = [
                            '#58a6ff', '#3fb9a0', '#f0883e', '#3fb950', '#bc8cff', '#f85149'
                        ];

                        function draw(t) {
                            animTime += 0.01;
                            var phase = Math.min(animTime / 2.0, 1.0);
                            var ease = phase * phase * (3 - 2 * phase);

                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var tokenY = 50;
                            var routerY = 160;
                            var expertY = 350;
                            var tokenSpacing = (W - 120) / numTokens;
                            var tokenStartX = 80;
                            var expertSpacing = (W - 160) / numExperts;
                            var expertStartX = 100;

                            // Draw section labels
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Input Tokens', 10, tokenY);
                            ctx.fillText('Router', 10, routerY);
                            ctx.fillText('Experts', 10, expertY);

                            // Draw tokens
                            for (var t2 = 0; t2 < numTokens; t2++) {
                                var tx = tokenStartX + t2 * tokenSpacing;
                                ctx.fillStyle = '#222255';
                                ctx.strokeStyle = viz.colors.blue + '88';
                                ctx.lineWidth = 1.5;
                                ctx.beginPath();
                                ctx.roundRect(tx - 22, tokenY - 16, 44, 32, 6);
                                ctx.fill();
                                ctx.stroke();

                                ctx.fillStyle = viz.colors.white;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText(tokenLabels[t2] || 'tok' + t2, tx, tokenY);
                            }

                            // Draw router box
                            ctx.fillStyle = '#1a1a40';
                            ctx.strokeStyle = viz.colors.purple + '88';
                            ctx.lineWidth = 1.5;
                            ctx.beginPath();
                            ctx.roundRect(40, routerY - 30, W - 80, 60, 8);
                            ctx.fill();
                            ctx.stroke();

                            ctx.fillStyle = viz.colors.purple;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Gating Network: softmax(W_g * x)  \u2192  select top-' + topK, W / 2, routerY - 10);

                            // Draw routing probabilities as small bars in router area
                            for (var t3 = 0; t3 < numTokens; t3++) {
                                var tx2 = tokenStartX + t3 * tokenSpacing;
                                var barW = 4;
                                var barMaxH = 22;
                                var barStartX = tx2 - (numExperts * barW) / 2;

                                for (var e = 0; e < numExperts; e++) {
                                    var bx = barStartX + e * barW;
                                    var bh = routerProbs[t3][e] * barMaxH;
                                    var isSelected = selectedExperts[t3].indexOf(e) !== -1;
                                    ctx.fillStyle = isSelected ? expertColors[e] : expertColors[e] + '33';
                                    ctx.fillRect(bx, routerY + 14 - bh, barW - 1, bh);
                                }

                                // Arrow from token to router
                                ctx.strokeStyle = viz.colors.text + '44';
                                ctx.lineWidth = 1;
                                ctx.beginPath();
                                ctx.moveTo(tx2, tokenY + 16);
                                ctx.lineTo(tx2, routerY - 30);
                                ctx.stroke();
                            }

                            // Draw experts
                            for (var e2 = 0; e2 < numExperts; e2++) {
                                var ex = expertStartX + e2 * expertSpacing;
                                ctx.fillStyle = expertColors[e2] + '22';
                                ctx.strokeStyle = expertColors[e2];
                                ctx.lineWidth = 2;
                                ctx.beginPath();
                                ctx.roundRect(ex - 30, expertY - 25, 60, 50, 8);
                                ctx.fill();
                                ctx.stroke();

                                ctx.fillStyle = expertColors[e2];
                                ctx.font = 'bold 11px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                ctx.fillText('FFN_' + (e2 + 1), ex, expertY - 6);

                                // Count tokens routed here
                                var count = 0;
                                for (var t4 = 0; t4 < numTokens; t4++) {
                                    if (selectedExperts[t4].indexOf(e2) !== -1) count++;
                                }
                                ctx.fillStyle = viz.colors.text;
                                ctx.font = '10px -apple-system,sans-serif';
                                ctx.fillText(count + ' tokens', ex, expertY + 14);
                            }

                            // Draw routing lines (token -> selected experts)
                            for (var t5 = 0; t5 < numTokens; t5++) {
                                var tx3 = tokenStartX + t5 * tokenSpacing;
                                for (var s = 0; s < selectedExperts[t5].length; s++) {
                                    var expIdx = selectedExperts[t5][s];
                                    var ex2 = expertStartX + expIdx * expertSpacing;
                                    var weight = routerProbs[t5][expIdx];

                                    // Animated line
                                    var lineProgress = Math.min(ease, 1.0);
                                    var startPt = { x: tx3, y: routerY + 30 };
                                    var endPt = { x: ex2, y: expertY - 25 };
                                    var curEnd = {
                                        x: startPt.x + (endPt.x - startPt.x) * lineProgress,
                                        y: startPt.y + (endPt.y - startPt.y) * lineProgress
                                    };

                                    ctx.strokeStyle = expertColors[expIdx] + '88';
                                    ctx.lineWidth = 1 + weight * 3;
                                    ctx.beginPath();
                                    ctx.moveTo(startPt.x, startPt.y);
                                    ctx.lineTo(curEnd.x, curEnd.y);
                                    ctx.stroke();

                                    // Weight label at midpoint
                                    if (lineProgress > 0.9) {
                                        var midX = (startPt.x + endPt.x) / 2;
                                        var midY = (startPt.y + endPt.y) / 2;
                                        ctx.fillStyle = expertColors[expIdx];
                                        ctx.font = '9px monospace';
                                        ctx.textAlign = 'center';
                                        ctx.textBaseline = 'middle';
                                        ctx.fillText(weight.toFixed(2), midX + 10, midY);
                                    }
                                }
                            }

                            // Bottom legend
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'bottom';
                            ctx.fillText('Each token is routed to top-' + topK + ' experts. Line thickness = gating weight. Small bars show softmax distribution over experts.', W / 2, H - 8);

                            // Load balance indicator
                            var loads = [];
                            for (var e3 = 0; e3 < numExperts; e3++) {
                                var load = 0;
                                for (var t6 = 0; t6 < numTokens; t6++) {
                                    if (selectedExperts[t6].indexOf(e3) !== -1) load++;
                                }
                                loads.push(load);
                            }
                            var maxLoad = Math.max.apply(null, loads);
                            var minLoad = Math.min.apply(null, loads);
                            var balanced = maxLoad - minLoad <= 2;

                            ctx.fillStyle = balanced ? viz.colors.green : viz.colors.yellow;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'right';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Load balance: ' + (balanced ? 'Good' : 'Uneven'), W - 20, 10);
                        }

                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'A Mixture of Experts model has \\(E = 64\\) experts and uses top-\\(k = 2\\) routing. Each expert is an FFN with 100M parameters, and the shared (non-expert) parameters total 500M. What are the total parameters? What fraction of expert parameters is activated per token?',
                    hint: 'Total expert params = \\(E \\times\\) params per expert. Active per token = \\(k\\) experts.',
                    solution: 'Total expert parameters: \\(64 \\times 100\\text{M} = 6.4\\text{B}\\). Total model parameters: \\(6.4\\text{B} + 500\\text{M} = 6.9\\text{B}\\). Per token, \\(k = 2\\) experts are activated, using \\(2 \\times 100\\text{M} = 200\\text{M}\\) expert parameters, which is \\(200\\text{M} / 6.4\\text{B} = 3.125\\%\\) of the total expert parameters.'
                },
                {
                    question: 'Why is load balancing critical in MoE training? What would happen if all tokens were routed to the same expert?',
                    hint: 'Think about both capacity utilization and, in distributed settings, compute bottlenecks.',
                    solution: 'If all tokens route to one expert, the other \\(E - 1\\) experts remain untrained and waste parameters. The popular expert becomes a bottleneck: it must process all tokens, negating the computational savings of sparse routing. In distributed training, where experts live on different devices, the overloaded device stalls all others (since the forward pass waits for the slowest expert). Additionally, the single expert lacks the capacity of the full model, so quality degrades. The auxiliary load balancing loss prevents this by penalizing uneven token distributions across experts.'
                },
                {
                    question: 'Compare the memory and compute characteristics of a dense 7B-parameter Transformer and a MoE model with 8 experts, top-2 routing, where each expert has 7B parameters in the FFN layer (like Mixtral 8x7B). Which uses more memory? Which uses more compute per token?',
                    hint: 'Memory depends on total parameters. Compute per token depends on active parameters.',
                    solution: 'The dense 7B model has 7B parameters and activates all of them per token. Mixtral 8x7B has approximately 47B total parameters (8 expert FFNs plus shared attention layers), so it requires roughly 6-7x more memory. However, per-token compute is approximately 13B active parameters (2 expert FFNs plus shared layers), roughly 2x the dense 7B model. So the MoE model uses much more memory but only moderately more compute per token, while having much more total capacity. The trade-off is: MoE is better when memory is plentiful but you need fast inference (low per-token FLOPS).'
                }
            ]
        },

        // ============================================================
        // Section 5: State Space Models & Mamba
        // ============================================================
        {
            id: 'state-space-models',
            title: 'State Space Models & Mamba',
            content: `
                <h2>State Space Models &amp; Mamba</h2>

                <div class="env-block intuition">
                    <div class="env-title">The Quadratic Bottleneck</div>
                    <div class="env-body"><p>The Transformer's self-attention mechanism has a fundamental limitation: its time and memory complexity scale as \\(O(N^2)\\) in sequence length \\(N\\). For a sequence of 100k tokens, the attention matrix alone has 10 billion entries. State Space Models (SSMs) offer an alternative that processes sequences in \\(O(N)\\) time by maintaining a fixed-size hidden state, much like RNNs, but with crucial innovations that make them practical and performant.</p></div>
                </div>

                <h3>Continuous-Time State Space Models</h3>

                <p>SSMs originate from control theory. A linear time-invariant (LTI) system maps an input signal \\(u(t)\\) to an output \\(y(t)\\) via a hidden state \\(x(t) \\in \\mathbb{R}^N\\):</p>

                \\[\\frac{dx}{dt} = \\mathbf{A} x(t) + \\mathbf{B} u(t)\\]
                \\[y(t) = \\mathbf{C} x(t) + D u(t)\\]

                <p>where \\(\\mathbf{A} \\in \\mathbb{R}^{N \\times N}\\), \\(\\mathbf{B} \\in \\mathbb{R}^{N \\times 1}\\), \\(\\mathbf{C} \\in \\mathbb{R}^{1 \\times N}\\), and \\(D \\in \\mathbb{R}\\). The state size \\(N\\) controls the model's "memory capacity."</p>

                <h3>Discretization</h3>

                <p>To apply this to discrete sequences, we discretize using a step size \\(\\Delta\\). The zero-order hold (ZOH) discretization gives:</p>

                \\[\\bar{\\mathbf{A}} = \\exp(\\Delta \\mathbf{A})\\]
                \\[\\bar{\\mathbf{B}} = (\\Delta \\mathbf{A})^{-1}(\\exp(\\Delta \\mathbf{A}) - \\mathbf{I}) \\cdot \\Delta \\mathbf{B}\\]

                <p>The discrete recurrence is:</p>

                \\[x_k = \\bar{\\mathbf{A}} x_{k-1} + \\bar{\\mathbf{B}} u_k\\]
                \\[y_k = \\mathbf{C} x_k\\]

                <div class="env-block definition">
                    <div class="env-title">Definition (Structured State Space Model, S4)</div>
                    <div class="env-body">
                        <p>The <strong>Structured State Space</strong> (S4) model (Gu et al., 2022) parameterizes \\(\\mathbf{A}\\) as a structured matrix (specifically, a diagonal plus low-rank matrix related to the HiPPO initialization). This structure enables:</p>
                        <ol>
                            <li><strong>Recurrent mode</strong>: process tokens one by one in \\(O(N)\\) per step, ideal for autoregressive generation.</li>
                            <li><strong>Convolutional mode</strong>: unroll the recurrence into a global convolution \\(y = \\bar{K} * u\\) where \\(\\bar{K} = (\\mathbf{C}\\bar{\\mathbf{B}}, \\mathbf{C}\\bar{\\mathbf{A}}\\bar{\\mathbf{B}}, \\mathbf{C}\\bar{\\mathbf{A}}^2\\bar{\\mathbf{B}}, \\dots)\\), computed in \\(O(L \\log L)\\) via FFT. This is ideal for parallel training.</li>
                        </ol>
                    </div>
                </div>

                <div class="env-block remark">
                    <div class="env-title">The Duality of Computation</div>
                    <div class="env-body"><p>The key insight of S4 is that the same linear system can be computed either as a recurrence (sequential, \\(O(L)\\) total, \\(O(1)\\) per step) or as a convolution (parallel, \\(O(L \\log L)\\) total). During training, convolution mode enables GPU parallelism. During inference, recurrence mode avoids reprocessing the entire context for each new token. This is the fundamental advantage over Transformers, which must attend to all previous tokens at every generation step (KV-cache helps but does not eliminate the \\(O(L)\\) per-step cost).</p></div>
                </div>

                <h3>Mamba: Selective State Spaces</h3>

                <p>The original S4 is a <em>linear time-invariant</em> system: the matrices \\(\\mathbf{A}, \\mathbf{B}, \\mathbf{C}\\) are the same for every input token. This limits the model's ability to selectively focus on or ignore specific inputs. <strong>Mamba</strong> (Gu &amp; Dao, 2024) makes the SSM <em>input-dependent</em>:</p>

                \\[\\mathbf{B}_k = f_B(\\mathbf{x}_k), \\quad \\mathbf{C}_k = f_C(\\mathbf{x}_k), \\quad \\Delta_k = f_\\Delta(\\mathbf{x}_k)\\]

                <p>where \\(f_B, f_C, f_\\Delta\\) are simple linear projections of the input. By making \\(\\Delta_k\\) input-dependent, the model can control <em>how much</em> of the previous state to retain: a large \\(\\Delta\\) decays the state quickly (forget), while a small \\(\\Delta\\) preserves it (remember). This is analogous to the gating mechanism in LSTMs.</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (Selective State Space, Mamba)</div>
                    <div class="env-body">
                        <p>Mamba uses a <strong>selective scan</strong> algorithm:</p>
                        <p>\\[x_k = \\bar{\\mathbf{A}}_k x_{k-1} + \\bar{\\mathbf{B}}_k u_k\\]</p>
                        <p>\\[y_k = \\mathbf{C}_k x_k\\]</p>
                        <p>where \\(\\bar{\\mathbf{A}}_k\\) and \\(\\bar{\\mathbf{B}}_k\\) depend on the input \\(\\mathbf{x}_k\\). Because the system is now time-varying, the convolution trick no longer applies directly. Instead, Mamba uses a hardware-efficient parallel scan algorithm on GPUs, computing the recurrence in \\(O(L)\\) total work while maintaining GPU parallelism via work-efficient scan.</p>
                    </div>
                </div>

                <div class="env-block example">
                    <div class="env-title">Example (Mamba vs. Transformer on Language Modeling)</div>
                    <div class="env-body">
                        <p>Mamba-3B matches Transformer models twice its size on language modeling benchmarks. On long-context tasks (sequence lengths of 16k-1M tokens), Mamba's linear scaling gives it a decisive throughput advantage: at sequence length 64k, Mamba achieves 5x higher throughput than a comparable Transformer with FlashAttention-2.</p>
                    </div>
                </div>

                <div class="viz-placeholder" data-viz="viz-ssm-complexity"></div>

                <h3>Hybrid Architectures</h3>

                <p>In practice, the best results come from hybrid architectures that combine SSM layers with attention layers. Models like Jamba (AI21, 2024) and Zamba interleave Mamba blocks with attention blocks, getting the best of both worlds: linear scaling for most of the processing, with occasional attention layers for tasks that require precise recall or complex relational reasoning.</p>

                <div class="env-block intuition">
                    <div class="env-title">SSM vs. Attention: Complementary Strengths</div>
                    <div class="env-body"><p>SSMs compress the entire history into a fixed-size state vector. This is memory-efficient but lossy: the model cannot perform exact retrieval of a specific past token. Attention, by contrast, retains all past tokens and can perform exact lookup, but at quadratic cost. For tasks requiring long-range dependencies but not exact recall (e.g., language modeling, audio generation), SSMs excel. For tasks requiring precise retrieval (e.g., copying a substring from 10k tokens ago), attention is superior. Hybrid models allocate each task to the appropriate mechanism.</p></div>
                </div>

                <div class="env-block warning">
                    <div class="env-title">The Recall Gap</div>
                    <div class="env-body"><p>Pure SSM models struggle with "associative recall" tasks: given a key, retrieve the associated value from earlier in the sequence. This is because the fixed-size state vector cannot store an arbitrary number of key-value associations. Zoology (Arora et al., 2023) systematically showed that attention-free models fail on such tasks, motivating the hybrid approach. This is a fundamental limitation of any fixed-state model, not a deficiency of SSMs specifically.</p></div>
                </div>
            `,
            visualizations: [
                {
                    id: 'viz-ssm-complexity',
                    title: 'Sequence Length vs. Computation: SSM vs. Transformer',
                    description: 'Compare the computational cost scaling of Transformers (quadratic) vs. SSMs (linear) as sequence length grows. Drag the sequence length slider to see how the gap widens dramatically for long sequences.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, { width: 780, height: 420, scale: 1, originX: 0, originY: 0 });
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        var maxSeqLen = 32000;
                        var currentLen = 8000;

                        VizEngine.createSlider(controls, 'Max Seq Len (k)', 4, 128, maxSeqLen / 1000, 4, function(v) {
                            maxSeqLen = v * 1000;
                        });

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            // Plot area
                            var plotLeft = 90;
                            var plotRight = W - 40;
                            var plotTop = 50;
                            var plotBottom = H - 70;
                            var plotW = plotRight - plotLeft;
                            var plotH = plotBottom - plotTop;

                            // Draw plot background
                            ctx.fillStyle = '#0e0e28';
                            ctx.fillRect(plotLeft, plotTop, plotW, plotH);
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 1;
                            ctx.strokeRect(plotLeft, plotTop, plotW, plotH);

                            // Grid lines
                            ctx.strokeStyle = viz.colors.grid + '44';
                            ctx.lineWidth = 0.5;
                            for (var gy = 0; gy <= 4; gy++) {
                                var yy = plotTop + (gy / 4) * plotH;
                                ctx.beginPath();
                                ctx.moveTo(plotLeft, yy);
                                ctx.lineTo(plotRight, yy);
                                ctx.stroke();
                            }
                            for (var gx = 0; gx <= 5; gx++) {
                                var xx = plotLeft + (gx / 5) * plotW;
                                ctx.beginPath();
                                ctx.moveTo(xx, plotTop);
                                ctx.lineTo(xx, plotBottom);
                                ctx.stroke();
                            }

                            // Compute curves
                            var numPoints = 200;
                            var transformerCost = function(n) { return n * n; }; // O(N^2)
                            var ssmCost = function(n) { return n * Math.log(n + 1); }; // O(N log N) for S4, or just N for Mamba
                            var ssmLinear = function(n) { return n; }; // O(N) pure linear

                            // Find max cost for scaling
                            var maxCost = transformerCost(maxSeqLen);

                            // Draw Transformer curve (quadratic)
                            ctx.strokeStyle = viz.colors.red;
                            ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            for (var i = 0; i <= numPoints; i++) {
                                var n = (i / numPoints) * maxSeqLen;
                                var cost = transformerCost(n);
                                var px = plotLeft + (i / numPoints) * plotW;
                                var py = plotBottom - (cost / maxCost) * plotH;
                                if (py < plotTop) py = plotTop;
                                if (i === 0) ctx.moveTo(px, py);
                                else ctx.lineTo(px, py);
                            }
                            ctx.stroke();

                            // Draw S4 curve (N log N)
                            ctx.strokeStyle = viz.colors.yellow;
                            ctx.lineWidth = 2.5;
                            ctx.setLineDash([6, 4]);
                            ctx.beginPath();
                            for (var i2 = 0; i2 <= numPoints; i2++) {
                                var n2 = (i2 / numPoints) * maxSeqLen;
                                var cost2 = ssmCost(n2);
                                var px2 = plotLeft + (i2 / numPoints) * plotW;
                                var py2 = plotBottom - (cost2 / maxCost) * plotH;
                                if (py2 < plotTop) py2 = plotTop;
                                if (i2 === 0) ctx.moveTo(px2, py2);
                                else ctx.lineTo(px2, py2);
                            }
                            ctx.stroke();
                            ctx.setLineDash([]);

                            // Draw Mamba curve (linear)
                            ctx.strokeStyle = viz.colors.green;
                            ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            for (var i3 = 0; i3 <= numPoints; i3++) {
                                var n3 = (i3 / numPoints) * maxSeqLen;
                                var cost3 = ssmLinear(n3);
                                var px3 = plotLeft + (i3 / numPoints) * plotW;
                                var py3 = plotBottom - (cost3 / maxCost) * plotH;
                                if (py3 < plotTop) py3 = plotTop;
                                if (i3 === 0) ctx.moveTo(px3, py3);
                                else ctx.lineTo(px3, py3);
                            }
                            ctx.stroke();

                            // Axis labels
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Sequence Length N', (plotLeft + plotRight) / 2, plotBottom + 10);

                            // X axis ticks
                            ctx.font = '10px -apple-system,sans-serif';
                            for (var gx2 = 0; gx2 <= 5; gx2++) {
                                var val = Math.round(maxSeqLen * gx2 / 5 / 1000);
                                var xPos = plotLeft + (gx2 / 5) * plotW;
                                ctx.fillText(val + 'k', xPos, plotBottom + 25);
                            }

                            // Y axis label
                            ctx.save();
                            ctx.translate(20, (plotTop + plotBottom) / 2);
                            ctx.rotate(-Math.PI / 2);
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Computation Cost (relative)', 0, 0);
                            ctx.restore();

                            // Title
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 14px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Computational Scaling: Transformer vs. State Space Models', W / 2, 8);

                            // Legend
                            var legendX = plotLeft + 20;
                            var legendY = plotTop + 15;

                            ctx.fillStyle = viz.colors.red;
                            ctx.fillRect(legendX, legendY, 20, 3);
                            ctx.fillStyle = viz.colors.red;
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'middle';
                            ctx.fillText('Transformer: O(N\u00B2)', legendX + 28, legendY + 2);

                            ctx.strokeStyle = viz.colors.yellow;
                            ctx.lineWidth = 2;
                            ctx.setLineDash([4, 3]);
                            ctx.beginPath();
                            ctx.moveTo(legendX, legendY + 22);
                            ctx.lineTo(legendX + 20, legendY + 22);
                            ctx.stroke();
                            ctx.setLineDash([]);
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.fillText('S4: O(N log N)', legendX + 28, legendY + 22);

                            ctx.fillStyle = viz.colors.green;
                            ctx.fillRect(legendX, legendY + 40, 20, 3);
                            ctx.fillStyle = viz.colors.green;
                            ctx.fillText('Mamba: O(N)', legendX + 28, legendY + 42);

                            // Ratio callout at rightmost point
                            var ratioTransformer = transformerCost(maxSeqLen);
                            var ratioMamba = ssmLinear(maxSeqLen);
                            var ratio = Math.round(ratioTransformer / ratioMamba);

                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'bottom';
                            ctx.fillText('At N=' + (maxSeqLen/1000) + 'k: Transformer is ' + ratio.toLocaleString() + 'x more expensive than Mamba', W / 2, plotBottom + 55);

                            // Inference mode comparison on right side
                            var infoX = plotRight - 180;
                            var infoY = plotTop + plotH * 0.35;
                            ctx.fillStyle = viz.colors.bg + 'dd';
                            ctx.fillRect(infoX, infoY, 175, 110);
                            ctx.strokeStyle = viz.colors.grid;
                            ctx.lineWidth = 1;
                            ctx.strokeRect(infoX, infoY, 175, 110);

                            ctx.fillStyle = viz.colors.teal;
                            ctx.font = 'bold 11px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            ctx.fillText('Generation (per token):', infoX + 8, infoY + 8);

                            ctx.fillStyle = viz.colors.red;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillText('Transformer: O(N) with KV-cache', infoX + 8, infoY + 28);

                            ctx.fillStyle = viz.colors.green;
                            ctx.fillText('Mamba: O(1) per step', infoX + 8, infoY + 44);

                            ctx.fillStyle = viz.colors.teal;
                            ctx.font = 'bold 11px -apple-system,sans-serif';
                            ctx.fillText('Memory:', infoX + 8, infoY + 65);

                            ctx.fillStyle = viz.colors.red;
                            ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillText('Transformer: O(N) KV-cache', infoX + 8, infoY + 82);

                            ctx.fillStyle = viz.colors.green;
                            ctx.fillText('Mamba: O(1) state', infoX + 8, infoY + 96);
                        }

                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],
            exercises: [
                {
                    question: 'A Transformer with sequence length \\(N = 32{,}768\\) and hidden dimension \\(d = 4096\\) stores a KV-cache of size \\(O(N \\cdot d)\\) per layer. A Mamba model with state dimension \\(S = 16\\) stores a hidden state of size \\(O(d \\cdot S)\\). Compare the memory requirements for these two models at this sequence length (assume both have the same \\(d\\)).',
                    hint: 'For the Transformer, the KV-cache stores \\(2 \\times N \\times d\\) values per layer (keys and values). For Mamba, the hidden state is \\(d \\times S\\), independent of sequence length.',
                    solution: 'Transformer KV-cache per layer: \\(2 \\times 32{,}768 \\times 4{,}096 = 268{,}435{,}456\\) values (about 268M). Mamba state per layer: \\(4{,}096 \\times 16 = 65{,}536\\) values (about 65K). The ratio is \\(268{,}435{,}456 / 65{,}536 = 4{,}096\\). The Transformer KV-cache requires roughly 4000x more memory than the Mamba hidden state at this sequence length. At \\(N = 1M\\), the ratio would be \\(\\sim 128{,}000\\)x. This is the fundamental memory advantage of SSMs for long-context inference.'
                },
                {
                    question: 'Why does making the SSM parameters input-dependent (as in Mamba) break the ability to use convolution-mode computation? What alternative does Mamba use?',
                    hint: 'The convolution kernel \\(\\bar{K} = (\\mathbf{C}\\bar{\\mathbf{B}}, \\mathbf{C}\\bar{\\mathbf{A}}\\bar{\\mathbf{B}}, \\dots)\\) is precomputable only if \\(\\mathbf{A}, \\mathbf{B}, \\mathbf{C}\\) are constant across time steps.',
                    solution: 'The convolution representation relies on the system being <em>linear time-invariant</em> (LTI): the same matrices \\(\\mathbf{A}, \\mathbf{B}, \\mathbf{C}\\) apply at every time step, so the kernel \\(\\bar{K}_t = \\mathbf{C}\\bar{\\mathbf{A}}^t \\bar{\\mathbf{B}}\\) can be precomputed once and applied via FFT. When these matrices change at each step (as in Mamba), the kernel at step \\(t\\) depends on the product \\(\\bar{\\mathbf{A}}_t \\bar{\\mathbf{A}}_{t-1} \\cdots \\bar{\\mathbf{A}}_1\\), which varies for every position. No fixed convolution kernel exists. Instead, Mamba uses a <strong>parallel scan</strong> (also called prefix sum) algorithm. This computes the sequential recurrence in \\(O(L)\\) work with \\(O(\\log L)\\) depth on a GPU, achieving near-linear wall-clock time despite the sequential dependencies.'
                },
                {
                    question: 'Explain why pure SSM models struggle with "associative recall" tasks (e.g., given a sequence of key-value pairs followed by a query key, retrieve the associated value). What fundamental limitation is at play?',
                    hint: 'Consider the information bottleneck of compressing the entire sequence into a fixed-size state vector.',
                    solution: 'An SSM compresses the entire sequence history into a fixed-size state vector \\(x_k \\in \\mathbb{R}^{S}\\). To perform associative recall among \\(M\\) key-value pairs, the model must simultaneously store all \\(M\\) associations in this fixed-size state. By information-theoretic arguments, a state of size \\(S\\) can store at most \\(O(S)\\) bits of information. When \\(M\\) exceeds what the state can represent, retrieval fails. Attention, by contrast, stores all past tokens explicitly in the KV-cache and can perform exact lookup via the softmax mechanism, regardless of how many key-value pairs exist. This is the fundamental recall gap: fixed-state models compress and thus lose information, while attention retains everything at the cost of \\(O(N)\\) memory.'
                }
            ]
        }
    ]
});
