��
��
9
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
.
Identity

input"T
output"T"	
Ttype
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
�
MaxPool

input"T
output"T"
Ttype0:
2		"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
<
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
A
Relu
features"T
activations"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
H
ShardedFilename
basename	
shard

num_shards
filename
8
Softmax
logits"T
softmax"T"
Ttype:
2
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �"serve*1.3.02
b'unknown'��

global_step/Initializer/zerosConst*
value	B	 R *
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
�
global_step
VariableV2*
shape: *
dtype0	*
	container *
shared_name *
_class
loc:@global_step*
_output_shapes
: 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
validate_shape(*
use_locking(*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
inputPlaceholder*
dtype0*
shape:����������*(
_output_shapes
:����������
f
Reshape/shapeConst*%
valueB"����         *
dtype0*
_output_shapes
:
p
ReshapeReshapeinputReshape/shape*
T0*
Tshape0*/
_output_shapes
:���������
�
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"             *
dtype0* 
_class
loc:@conv2d/kernel*
_output_shapes
:
�
,conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *n���*
dtype0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
�
,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *n��=*
dtype0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
�
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*

seed*
seed2
*
dtype0*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
�
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
�
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
�
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
�
conv2d/kernel
VariableV2*
shape: *
dtype0*
	container *
shared_name * 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
�
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
�
conv2d/kernel/readIdentityconv2d/kernel*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
�
conv2d/bias/Initializer/zerosConst*
valueB *    *
dtype0*
_class
loc:@conv2d/bias*
_output_shapes
: 
�
conv2d/bias
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_class
loc:@conv2d/bias*
_output_shapes
: 
�
conv2d/bias/AssignAssignconv2d/biasconv2d/bias/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@conv2d/bias*
_output_shapes
: 
n
conv2d/bias/readIdentityconv2d/bias*
T0*
_class
loc:@conv2d/bias*
_output_shapes
: 
q
conv2d/convolution/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
q
 conv2d/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d/convolutionConv2DReshapeconv2d/kernel/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*/
_output_shapes
:��������� 
�
conv2d/BiasAddBiasAddconv2d/convolutionconv2d/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:��������� 
]
conv2d/ReluReluconv2d/BiasAdd*
T0*/
_output_shapes
:��������� 
�
max_pooling2d/MaxPoolMaxPoolconv2d/Relu*
T0*
ksize
*
strides
*
paddingVALID*
data_formatNHWC*/
_output_shapes
:��������� 
�
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"          @   *
dtype0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
:
�
.conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *��L�*
dtype0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *��L=*
dtype0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*

seed*
seed2*
dtype0*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @
�
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @
�
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @
�
conv2d_1/kernel
VariableV2*
shape: @*
dtype0*
	container *
shared_name *"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @
�
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @
�
conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @
�
conv2d_1/bias/Initializer/zerosConst*
valueB@*    *
dtype0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
�
conv2d_1/bias
VariableV2*
shape:@*
dtype0*
	container *
shared_name * 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
�
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/bias/Initializer/zeros*
T0*
validate_shape(*
use_locking(* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
s
conv2d_2/convolution/ShapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
conv2d_2/convolutionConv2Dmax_pooling2d/MaxPoolconv2d_1/kernel/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������@
�
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:���������@
a
conv2d_2/ReluReluconv2d_2/BiasAdd*
T0*/
_output_shapes
:���������@
�
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu*
T0*
ksize
*
strides
*
paddingVALID*
data_formatNHWC*/
_output_shapes
:���������@
`
Reshape_1/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:

	Reshape_1Reshapemax_pooling2d_2/MaxPoolReshape_1/shape*
T0*
Tshape0*(
_output_shapes
:����������
�
-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"   �  *
dtype0*
_class
loc:@dense/kernel*
_output_shapes
:
�
+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *����*
dtype0*
_class
loc:@dense/kernel*
_output_shapes
: 
�
+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *���=*
dtype0*
_class
loc:@dense/kernel*
_output_shapes
: 
�
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*

seed*
seed24*
dtype0*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
��
�
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
�
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
��
�
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
��
�
dense/kernel
VariableV2*
shape:
��*
dtype0*
	container *
shared_name *
_class
loc:@dense/kernel* 
_output_shapes
:
��
�
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense/kernel* 
_output_shapes
:
��
w
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
��
�
dense/bias/Initializer/zerosConst*
valueB�*    *
dtype0*
_class
loc:@dense/bias*
_output_shapes	
:�
�

dense/bias
VariableV2*
shape:�*
dtype0*
	container *
shared_name *
_class
loc:@dense/bias*
_output_shapes	
:�
�
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense/bias*
_output_shapes	
:�
l
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias*
_output_shapes	
:�
�
dense/MatMulMatMul	Reshape_1dense/kernel/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:����������
�
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
T

dense/ReluReludense/BiasAdd*
T0*(
_output_shapes
:����������
[
dropout/IdentityIdentity
dense/Relu*
T0*(
_output_shapes
:����������
�
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"�  
   *
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
:
�
-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *#޽*
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
�
-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *#�=*
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
�
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*

seed*
seed2F*
dtype0*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�

�
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
�
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�

�
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�

�
dense_1/kernel
VariableV2*
shape:	�
*
dtype0*
	container *
shared_name *!
_class
loc:@dense_1/kernel*
_output_shapes
:	�

�
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�

|
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�

�
dense_1/bias/Initializer/zerosConst*
valueB
*    *
dtype0*
_class
loc:@dense_1/bias*
_output_shapes
:

�
dense_1/bias
VariableV2*
shape:
*
dtype0*
	container *
shared_name *
_class
loc:@dense_1/bias*
_output_shapes
:

�
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_1/bias*
_output_shapes
:

q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:

�
dense_2/MatMulMatMuldropout/Identitydense_1/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:���������

�
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������

]
dense_2/SoftmaxSoftmaxdense_2/BiasAdd*
T0*'
_output_shapes
:���������

U
outputIdentitydense_2/Softmax*
T0*'
_output_shapes
:���������

P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_d75ce01604bf4b138448e0726fa0fe77/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�	Bconv2d/biasBconv2d/kernelBconv2d_1/biasBconv2d_1/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBglobal_step*
dtype0*
_output_shapes
:	
u
save/SaveV2/shape_and_slicesConst*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesconv2d/biasconv2d/kernelconv2d_1/biasconv2d_1/kernel
dense/biasdense/kerneldense_1/biasdense_1/kernelglobal_step*
dtypes
2		
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
T0*

axis *
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
o
save/RestoreV2/tensor_namesConst* 
valueBBconv2d/bias*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/AssignAssignconv2d/biassave/RestoreV2*
T0*
validate_shape(*
use_locking(*
_class
loc:@conv2d/bias*
_output_shapes
: 
s
save/RestoreV2_1/tensor_namesConst*"
valueBBconv2d/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1Assignconv2d/kernelsave/RestoreV2_1*
T0*
validate_shape(*
use_locking(* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
s
save/RestoreV2_2/tensor_namesConst*"
valueBBconv2d_1/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_2Assignconv2d_1/biassave/RestoreV2_2*
T0*
validate_shape(*
use_locking(* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
u
save/RestoreV2_3/tensor_namesConst*$
valueBBconv2d_1/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assignconv2d_1/kernelsave/RestoreV2_3*
T0*
validate_shape(*
use_locking(*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @
p
save/RestoreV2_4/tensor_namesConst*
valueBB
dense/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_4Assign
dense/biassave/RestoreV2_4*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense/bias*
_output_shapes	
:�
r
save/RestoreV2_5/tensor_namesConst*!
valueBBdense/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_5Assigndense/kernelsave/RestoreV2_5*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense/kernel* 
_output_shapes
:
��
r
save/RestoreV2_6/tensor_namesConst*!
valueBBdense_1/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6Assigndense_1/biassave/RestoreV2_6*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_1/bias*
_output_shapes
:

t
save/RestoreV2_7/tensor_namesConst*#
valueBBdense_1/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_7Assigndense_1/kernelsave/RestoreV2_7*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�

q
save/RestoreV2_8/tensor_namesConst* 
valueBBglobal_step*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2	*
_output_shapes
:
�
save/Assign_8Assignglobal_stepsave/RestoreV2_8*
T0	*
validate_shape(*
use_locking(*
_class
loc:@global_step*
_output_shapes
: 
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8
-
save/restore_allNoOp^save/restore_shard

initNoOp

init_all_tablesNoOp
+

group_depsNoOp^init^init_all_tables
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_e4cd6c65c9854507a5ff9b2a2dce2ea2/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
�
save_1/SaveV2/tensor_namesConst*�
value�B�	Bconv2d/biasBconv2d/kernelBconv2d_1/biasBconv2d_1/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBglobal_step*
dtype0*
_output_shapes
:	
w
save_1/SaveV2/shape_and_slicesConst*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
�
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesconv2d/biasconv2d/kernelconv2d_1/biasconv2d_1/kernel
dense/biasdense/kerneldense_1/biasdense_1/kernelglobal_step*
dtypes
2		
�
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
�
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
N*
T0*

axis *
_output_shapes
:
�
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
�
save_1/IdentityIdentitysave_1/Const^save_1/control_dependency^save_1/MergeV2Checkpoints*
T0*
_output_shapes
: 
q
save_1/RestoreV2/tensor_namesConst* 
valueBBconv2d/bias*
dtype0*
_output_shapes
:
j
!save_1/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save_1/AssignAssignconv2d/biassave_1/RestoreV2*
T0*
validate_shape(*
use_locking(*
_class
loc:@conv2d/bias*
_output_shapes
: 
u
save_1/RestoreV2_1/tensor_namesConst*"
valueBBconv2d/kernel*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save_1/Assign_1Assignconv2d/kernelsave_1/RestoreV2_1*
T0*
validate_shape(*
use_locking(* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
u
save_1/RestoreV2_2/tensor_namesConst*"
valueBBconv2d_1/bias*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save_1/RestoreV2_2	RestoreV2save_1/Constsave_1/RestoreV2_2/tensor_names#save_1/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save_1/Assign_2Assignconv2d_1/biassave_1/RestoreV2_2*
T0*
validate_shape(*
use_locking(* 
_class
loc:@conv2d_1/bias*
_output_shapes
:@
w
save_1/RestoreV2_3/tensor_namesConst*$
valueBBconv2d_1/kernel*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save_1/RestoreV2_3	RestoreV2save_1/Constsave_1/RestoreV2_3/tensor_names#save_1/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save_1/Assign_3Assignconv2d_1/kernelsave_1/RestoreV2_3*
T0*
validate_shape(*
use_locking(*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @
r
save_1/RestoreV2_4/tensor_namesConst*
valueBB
dense/bias*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save_1/RestoreV2_4	RestoreV2save_1/Constsave_1/RestoreV2_4/tensor_names#save_1/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save_1/Assign_4Assign
dense/biassave_1/RestoreV2_4*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense/bias*
_output_shapes	
:�
t
save_1/RestoreV2_5/tensor_namesConst*!
valueBBdense/kernel*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save_1/RestoreV2_5	RestoreV2save_1/Constsave_1/RestoreV2_5/tensor_names#save_1/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save_1/Assign_5Assigndense/kernelsave_1/RestoreV2_5*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense/kernel* 
_output_shapes
:
��
t
save_1/RestoreV2_6/tensor_namesConst*!
valueBBdense_1/bias*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save_1/RestoreV2_6	RestoreV2save_1/Constsave_1/RestoreV2_6/tensor_names#save_1/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save_1/Assign_6Assigndense_1/biassave_1/RestoreV2_6*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_1/bias*
_output_shapes
:

v
save_1/RestoreV2_7/tensor_namesConst*#
valueBBdense_1/kernel*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save_1/RestoreV2_7	RestoreV2save_1/Constsave_1/RestoreV2_7/tensor_names#save_1/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save_1/Assign_7Assigndense_1/kernelsave_1/RestoreV2_7*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�

s
save_1/RestoreV2_8/tensor_namesConst* 
valueBBglobal_step*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save_1/RestoreV2_8	RestoreV2save_1/Constsave_1/RestoreV2_8/tensor_names#save_1/RestoreV2_8/shape_and_slices*
dtypes
2	*
_output_shapes
:
�
save_1/Assign_8Assignglobal_stepsave_1/RestoreV2_8*
T0	*
validate_shape(*
use_locking(*
_class
loc:@global_step*
_output_shapes
: 
�
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"�
	variables��
7
global_step:0global_step/Assignglobal_step/read:0
=
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:0
7
conv2d/bias:0conv2d/bias/Assignconv2d/bias/read:0
C
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:0
=
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:0
:
dense/kernel:0dense/kernel/Assigndense/kernel/read:0
4
dense/bias:0dense/bias/Assigndense/bias/read:0
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0" 
global_step

global_step:0"�
trainable_variables��
=
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:0
7
conv2d/bias:0conv2d/bias/Assignconv2d/bias/read:0
C
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:0
=
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:0
:
dense/kernel:0dense/kernel/Assigndense/kernel/read:0
4
dense/bias:0dense/bias/Assigndense/bias/read:0
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0" 
legacy_init_op


group_deps*�
serving_defaults
+
inputKey
input:0����������(
score
output:0���������
tensorflow/serving/predict