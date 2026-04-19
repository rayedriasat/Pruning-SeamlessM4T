[P8] Total params      : 1039.1M
257.5s	163	[P8] Trainable (T2U)   : 182.0M  (17.5%)
257.5s	164	  GPU mem: 1.05 GB alloc / 1.06 GB reserved
257.7s	165	[P8] Loading teacher (facebook/seamless-m4t-v2-large)...
257.7s	166	Loading processor from facebook/seamless-m4t-v2-large...
263.9s	167	Loading model  -- may take 5-10 min...
299.3s	168	SeamlessM4Tv2ForSpeechToSpeech LOAD REPORT from: facebook/seamless-m4t-v2-large
299.3s	169	Key                                                      | Status     |  | 
299.3s	170	---------------------------------------------------------+------------+--+-
299.3s	171	text_encoder.layers.{0...23}.self_attn.q_proj.weight     | UNEXPECTED |  | 
299.3s	172	text_encoder.layers.{0...23}.self_attn.v_proj.bias       | UNEXPECTED |  | 
299.3s	173	text_encoder.layers.{0...23}.self_attn_layer_norm.bias   | UNEXPECTED |  | 
299.3s	174	text_encoder.layers.{0...23}.ffn_layer_norm.weight       | UNEXPECTED |  | 
299.3s	175	text_encoder.layers.{0...23}.ffn.fc1.bias                | UNEXPECTED |  | 
299.3s	176	text_encoder.layers.{0...23}.self_attn.out_proj.bias     | UNEXPECTED |  | 
299.3s	177	text_encoder.layers.{0...23}.self_attn.k_proj.bias       | UNEXPECTED |  | 
299.3s	178	text_encoder.layers.{0...23}.ffn.fc2.weight              | UNEXPECTED |  | 
299.3s	179	text_encoder.layers.{0...23}.ffn_layer_norm.bias         | UNEXPECTED |  | 
299.3s	180	text_encoder.layers.{0...23}.self_attn_layer_norm.weight | UNEXPECTED |  | 
299.3s	181	text_encoder.layers.{0...23}.self_attn.v_proj.weight     | UNEXPECTED |  | 
299.3s	182	text_encoder.layers.{0...23}.ffn.fc1.weight              | UNEXPECTED |  | 
299.3s	183	text_encoder.layers.{0...23}.self_attn.k_proj.weight     | UNEXPECTED |  | 
299.3s	184	text_encoder.layers.{0...23}.self_attn.q_proj.bias       | UNEXPECTED |  | 
299.3s	185	text_encoder.layers.{0...23}.ffn.fc2.bias                | UNEXPECTED |  | 
299.3s	186	text_encoder.layers.{0...23}.self_attn.out_proj.weight   | UNEXPECTED |  | 
299.3s	187	text_encoder.layer_norm.weight                           | UNEXPECTED |  | 
299.3s	188	text_encoder.layer_norm.bias                             | UNEXPECTED |  | 
299.3s	189	
299.3s	190	Notes:
299.3s	191	- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
299.3s	192	
299.3s	193	SeamlessM4Tv2ForSpeechToSpeech LOAD REPORT from: facebook/seamless-m4t-v2-large
299.3s	194	Key                                                      | Status     |  | 
299.3s	195	---------------------------------------------------------+------------+--+-
299.3s	196	text_encoder.layers.{0...23}.self_attn.q_proj.weight     | UNEXPECTED |  | 
299.3s	197	text_encoder.layers.{0...23}.self_attn.v_proj.bias       | UNEXPECTED |  | 
299.3s	198	text_encoder.layers.{0...23}.self_attn_layer_norm.bias   | UNEXPECTED |  | 
299.3s	199	text_encoder.layers.{0...23}.ffn_layer_norm.weight       | UNEXPECTED |  | 
299.3s	200	text_encoder.layers.{0...23}.ffn.fc1.bias                | UNEXPECTED |  | 
299.3s	201	text_encoder.layers.{0...23}.self_attn.out_proj.bias     | UNEXPECTED |  | 
299.3s	202	text_encoder.layers.{0...23}.self_attn.k_proj.bias       | UNEXPECTED |  | 
299.3s	203	text_encoder.layers.{0...23}.ffn.fc2.weight              | UNEXPECTED |  | 
299.3s	204	text_encoder.layers.{0...23}.ffn_layer_norm.bias         | UNEXPECTED |  | 
299.3s	205	text_encoder.layers.{0...23}.self_attn_layer_norm.weight | UNEXPECTED |  | 
299.3s	206	text_encoder.layers.{0...23}.self_attn.v_proj.weight     | UNEXPECTED |  | 
299.3s	207	text_encoder.layers.{0...23}.ffn.fc1.weight              | UNEXPECTED |  | 
299.3s	208	text_encoder.layers.{0...23}.self_attn.k_proj.weight     | UNEXPECTED |  | 
299.3s	209	text_encoder.layers.{0...23}.self_attn.q_proj.bias       | UNEXPECTED |  | 
299.3s	210	text_encoder.layers.{0...23}.ffn.fc2.bias                | UNEXPECTED |  | 
299.3s	211	text_encoder.layers.{0...23}.self_attn.out_proj.weight   | UNEXPECTED |  | 
299.3s	212	text_encoder.layer_norm.weight                           | UNEXPECTED |  | 
299.3s	213	text_encoder.layer_norm.bias                             | UNEXPECTED |  | 
299.3s	214	
299.3s	215	Notes:
299.3s	216	- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
301.3s	217	Model loaded.
301.3s	218	  GPU mem: 2.89 GB alloc / 2.90 GB reserved
301.3s	219	[P8] Teacher params : 1805.5M
301.3s	220	[P8] Student params : 1039.1M
301.3s	221	  GPU mem: 2.89 GB alloc / 2.90 GB reserved
301.4s	222	[P8] KD loss helpers ready.
301.4s	223	     Temperature=2.0  Alpha=0.7  MaxSteps=500
301.5s	224	[ckpt] No checkpoint for 'phase8_kd'
301.5s	225	[P8] Starting KD from scratch.
301.5s	226	[P8] Optimiser: AdamW  LR=3e-05  EffectiveBatch=8
306.1s	227	  [ERR] Step 0: You have to specify either decoder_input_ids or decoder_inputs_embeds
306.8s	228	  [ERR] Step 0: You have to specify either decoder_input_ids or decoder_inputs_embeds
307.5s	229	  [ERR] Step 0: You have to specify either decoder_input_ids or decoder_inputs_embeds
308.2s	230	  [ERR] Step 0: You have to specify either decoder_input_ids or decoder_inputs_embeds
308.9s	231	  [ERR] Step 0: You have to specify either decoder_input_ids or decoder_inputs_embeds
309.6s	232	  [ERR] Step 0: You have to specify either decoder_input_ids or decoder_inputs_embeds
309.6s	233	[P8] CRITICAL: Too many consecutive errors. Stopping.
309.6s	234	
309.6s	235	[P8] KD complete. Final step: 0  Time: 0.1 min
309.6s	236	[ckpt] Saved phase8_kd_step000000.pt (0.0 MB)
313.9s	237	[P8] Training curve saved.
314.1s	238	  [config] sync done.
314.1s	239	[model] Saving phase8_kd → /kaggle/working/models/phase8_kd ...
314.1s	240	  [config] sync done.
314.1s	241	  Saved custom state: ['_vocab_remap_to_old']
314.1s	242	  Saved pruning_manifest.pt keys=['stage_name']
320.8s	243	[model] Local save done. 2110 MB in 8 files.
320.8s	244	[model] Pushing to rclone remote...