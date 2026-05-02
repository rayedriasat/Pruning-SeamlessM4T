Debug1:
teacher_text_sequences (NEW IDs): [3, 22690, 65, 234, 5913, 210, 3006, 83, 7548, 553, 2844, 2269, 1511, 20427, 541, 1239, 3475, 8655, 6671, 65, 4343, 12189, 172, 11100, 315, 5640, 2700, 20423, 3]
Shape: torch.Size([1, 29]), dtype: torch.int64
Decoded text: ['in the case of sending things by air freight, it may take several days in some ways to clear and customs.']

t2u_input_ids (NEW, positions 2:-1): [65, 234, 5913, 210, 3006, 83, 7548, 553, 2844, 2269, 1511, 20427, 541, 1239, 3475, 8655, 6671, 65, 4343, 12189] ...

Subwords from NEW ids (first 20): ['▁in', '▁the', '▁case', '▁of', '▁send', 'ing', '▁things', '▁by', '▁air', '▁fre', 'ight', ',', '▁it', '▁may', '▁take', '▁several', '▁days', '▁in', '▁some', '▁ways']
  None count: 0/26

Subwords from OLD ids (first 20): ['te', 'sa', 'None', '▁no', '▁molec', 'ic', 'None', 'con', '▁प्री', '▁আমাকে', '▁bach', 'None', '▁عام', 'ظم', '▁क्यू', 'None', 'None', 'te', '▁excurs', 'None']
  None count: 9/26

Subword agreement NEW vs OLD: 0/26 (0.0%)
PASS if ≥90%  |  FAIL (and char_count_per_id will be garbage) if low

Debug2:
[00] s_len= 230  t_len= 220  common= 220 (100%)  logit_shape_s=(1, 230, 10082)  logit_shape_t=(1, 220, 10082)
[01] s_len= 284  t_len= 301  common= 284 (94%)  logit_shape_s=(1, 284, 10082)  logit_shape_t=(1, 301, 10082)
[02] s_len= 390  t_len= 446  common= 390 (87%)  logit_shape_s=(1, 390, 10082)  logit_shape_t=(1, 446, 10082)
[03] s_len= 305  t_len= 335  common= 305 (91%)  logit_shape_s=(1, 305, 10082)  logit_shape_t=(1, 335, 10082)
[04] s_len= 197  t_len= 235  common= 197 (84%)  logit_shape_s=(1, 197, 10082)  logit_shape_t=(1, 235, 10082)
[05] s_len= 285  t_len= 370  common= 285 (77%)  logit_shape_s=(1, 285, 10082)  logit_shape_t=(1, 370, 10082)
[06] s_len= 231  t_len= 242  common= 231 (95%)  logit_shape_s=(1, 231, 10082)  logit_shape_t=(1, 242, 10082)
[07] s_len= 228  t_len= 240  common= 228 (95%)  logit_shape_s=(1, 228, 10082)  logit_shape_t=(1, 240, 10082)
[08] s_len= 421  t_len= 494  common= 421 (85%)  logit_shape_s=(1, 421, 10082)  logit_shape_t=(1, 494, 10082)
[09] s_len= 219  t_len= 230  common= 219 (95%)  logit_shape_s=(1, 219, 10082)  logit_shape_t=(1, 230, 10082)
[10] s_len= 281  t_len= 273  common= 273 (100%)  logit_shape_s=(1, 281, 10082)  logit_shape_t=(1, 273, 10082)
[11] s_len= 267  t_len= 285  common= 267 (94%)  logit_shape_s=(1, 267, 10082)  logit_shape_t=(1, 285, 10082)
[12] s_len= 260  t_len= 261  common= 260 (100%)  logit_shape_s=(1, 260, 10082)  logit_shape_t=(1, 261, 10082)
[13] s_len= 283  t_len= 360  common= 283 (79%)  logit_shape_s=(1, 283, 10082)  logit_shape_t=(1, 360, 10082)
[14] s_len= 298  t_len= 312  common= 298 (96%)  logit_shape_s=(1, 298, 10082)  logit_shape_t=(1, 312, 10082)
[15] s_len= 386  t_len= 364  common= 364 (100%)  logit_shape_s=(1, 386, 10082)  logit_shape_t=(1, 364, 10082)
[16] s_len= 317  t_len= 312  common= 312 (100%)  logit_shape_s=(1, 317, 10082)  logit_shape_t=(1, 312, 10082)
[17] s_len= 388  t_len= 394  common= 388 (98%)  logit_shape_s=(1, 388, 10082)  logit_shape_t=(1, 394, 10082)
[18] s_len= 257  t_len= 337  common= 257 (76%)  logit_shape_s=(1, 257, 10082)  logit_shape_t=(1, 337, 10082)
[19] s_len= 277  t_len= 284  common= 277 (98%)  logit_shape_s=(1, 277, 10082)  logit_shape_t=(1, 284, 10082)

Summary over 20 samples:
  student_len : mean=290.2  min=197  max=421
  teacher_len : mean=314.8  min=220  max=494
  common_len  : mean=287.9  min=197  max=421
  coverage    : 91.5% of teacher sequence trained per step


Debug3:
=== Unit sequence snapshot (run once now, run again after 50 steps) ===

[Sample 0] tgt=ben  unit_len=317
  units (first 30): [8980, 7167, 3660, 8303, 1311, 5832, 5832, 5832, 9976, 4922, 310, 9730, 7681, 1059, 5533, 6323, 1862, 5920, 5920, 6151, 6151, 6151, 664, 5319, 3033, 4932, 3561, 6779, 6779, 6779]
  top-5 units: [(5103, 10), (1306, 8), (7252, 8), (8504, 7), (6279, 7)]
  unique units: 129  entropy=6.70
  waveform shape=(101440,)  rms=0.0875  max=0.7764

[Sample 1] tgt=ben  unit_len=237
  units (first 30): [5729, 7167, 3660, 8303, 6233, 6233, 5099, 5429, 1629, 5665, 2164, 2422, 6417, 1634, 5489, 5342, 4816, 4816, 9111, 8404, 8404, 1629, 430, 2164, 2164, 8901, 3202, 7239, 7239, 2427]
  top-5 units: [(1402, 8), (5064, 7), (5198, 7), (8791, 7), (2511, 6)]
  unique units: 112  entropy=6.50
  waveform shape=(76160,)  rms=0.0965  max=0.9053

[Sample 2] tgt=ben  unit_len=286
  units (first 30): [8980, 7167, 3660, 7483, 729, 8303, 3127, 3127, 5350, 1486, 1486, 512, 512, 7809, 8694, 8191, 9545, 9545, 4168, 2871, 2871, 8324, 6303, 6135, 6135, 6135, 3322, 287, 287, 5142]
  top-5 units: [(8791, 6), (1306, 5), (845, 5), (2385, 5), (1309, 5)]
  unique units: 136  entropy=6.87
  waveform shape=(91520,)  rms=0.0982  max=0.8403

[Sample 3] tgt=ben  unit_len=393
  units (first 30): [8980, 7167, 3660, 729, 4835, 4835, 5148, 5734, 5734, 1444, 9524, 8392, 8593, 2627, 9717, 9717, 5304, 5304, 2548, 2548, 8009, 838, 838, 3943, 3943, 2078, 2078, 6876, 3901, 5019]
  top-5 units: [(7729, 18), (5286, 9), (8611, 8), (5055, 8), (793, 7)]
  unique units: 163  entropy=6.98
  waveform shape=(125760,)  rms=0.0955  max=0.8184

[Sample 4] tgt=ben  unit_len=249
  units (first 30): [8980, 7167, 1580, 1580, 8212, 8212, 8212, 1319, 8022, 8022, 6019, 6019, 1254, 8177, 5225, 2088, 2088, 3236, 4433, 2529, 6317, 8924, 8924, 5319, 5319, 3033, 52, 52, 52, 7607]
  top-5 units: [(2222, 8), (1306, 7), (3594, 7), (793, 6), (9240, 5)]
  unique units: 120  entropy=6.66
  waveform shape=(80000,)  rms=0.1096  max=0.8818


Debug4:
 T2U mode: full native training
  Cast 0 trainable FP16 params to FP32
loss=2.7214  soft=0.9693  hard=7.0365  len=0.2888

Component                                        grad_norm    param_norm    n_params
-------------------------------------------------------------------------------------
  decoder                                           3.5042      322.5567  115,954,690
  encoder                                           0.6194      316.9336  83,957,760


Debug5:
Logit shape check:
  student:  (1, 304, 10082)  — expected last dim=10082
  teacher:  (1, 288, 10082)  — expected last dim=10082
  PASS: last dim is unit vocab

Teacher argmax units (first 20): [8980, 6315, 3103, 3103, 3103, 1871, 8422, 7329, 4222, 4222, 3023, 3023, 766, 766, 1120, 1120, 636, 1833, 1833, 2712]
Cached teacher units (first 20): [8980, 6315, 3103, 3103, 3103, 1871, 8422, 7329, 4222, 4222, 3023, 3023, 766, 766, 1120, 1120, 636, 1833, 1833, 2712]
Teacher argmax vs cached match rate: 100.00%
  → Low match rate means teacher logits don't reproduce the cached unit sequence

Student argmax units (first 20): [5729, 7167, 3103, 3103, 3103, 1871, 7329, 7329, 4222, 3023, 3023, 3023, 766, 1120, 1833, 1833, 1833, 2712, 2712, 2712]
Student vs teacher argmax agreement on common prefix: 5.21%  (over 288 common positions)

student: softmax entropy=7.063  unit token entropy=5.596
  (near-zero softmax entropy = collapsed logits = training target is a 1-hot spike)

teacher: softmax entropy=3.150  unit token entropy=7.420
  (near-zero softmax entropy = collapsed logits = training target is a 1-hot spike)