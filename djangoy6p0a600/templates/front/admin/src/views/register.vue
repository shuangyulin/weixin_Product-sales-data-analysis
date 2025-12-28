<template>
	<div>
		<div class="register-container">
			<el-form v-if="pageFlag=='register'" ref="ruleForm" class="rgs-form animate__animated animate__" :model="ruleForm" :rules="rules">
				<div class="rgs-form2">
					<div class="title">基于微信小程序的直播带货商品数据分析系统的设计与实现</div>
					<el-form-item class="list-item" v-if="tableName=='yonghu'">
						<div class="lable" :class="changeRules('yonghuzhanghao')?'required':''">用户账号：</div>
						<el-input  v-model="ruleForm.yonghuzhanghao"  autocomplete="off" placeholder="用户账号"  type="text"  />
					</el-form-item>
					<el-form-item class="list-item" v-if="tableName=='yonghu'">
						<div class="lable" :class="changeRules('mima')?'required':''">密码：</div>
						<el-input  v-model="ruleForm.mima"  autocomplete="off" placeholder="密码"  type="password"  />
					</el-form-item>
					<el-form-item class="list-item" v-if="tableName=='yonghu'">
						<div class="lable" :class="changeRules('mima')?'required':''">确认密码：</div>
						<el-input  v-model="ruleForm.mima2" autocomplete="off" placeholder="确认密码" type="password" />
					</el-form-item>
					<el-form-item class="list-item" v-if="tableName=='yonghu'">
						<div class="lable" :class="changeRules('yonghuxingming')?'required':''">用户姓名：</div>
						<el-input  v-model="ruleForm.yonghuxingming"  autocomplete="off" placeholder="用户姓名"  type="text"  />
					</el-form-item>
					<el-form-item class="list-item" v-if="tableName=='yonghu'">
						<div class="lable" :class="changeRules('xingbie')?'required':''">性别：</div>
						<el-select v-model="ruleForm.xingbie" placeholder="请选择性别" >
							<el-option
								v-for="(item,index) in yonghuxingbieOptions"
								v-bind:key="index"
								:label="item"
								:value="item">
							</el-option>
						</el-select>
					</el-form-item>
					<el-form-item class="list-item" v-if="tableName=='yonghu'">
						<div class="lable" :class="changeRules('shouji')?'required':''">手机：</div>
						<el-input  v-model="ruleForm.shouji"  autocomplete="off" placeholder="手机"  type="text"  />
					</el-form-item>
					<el-form-item class="list-item" v-if="tableName=='yonghu'">
						<div class="lable" :class="changeRules('touxiang')?'required':''">头像：</div>
						<file-upload
							tip="点击上传头像"
							action="file/upload"
							:limit="3"
							:multiple="true"
							:fileUrls="ruleForm.touxiang?ruleForm.touxiang:''"
							@change="yonghutouxiangUploadChange"
						></file-upload>
					</el-form-item>
					<el-form-item class="list-item" v-if="tableName=='shangjia'">
						<div class="lable" :class="changeRules('shangjiazhanghao')?'required':''">商家账号：</div>
						<el-input  v-model="ruleForm.shangjiazhanghao"  autocomplete="off" placeholder="商家账号"  type="text"  />
					</el-form-item>
					<el-form-item class="list-item" v-if="tableName=='shangjia'">
						<div class="lable" :class="changeRules('mima')?'required':''">密码：</div>
						<el-input  v-model="ruleForm.mima"  autocomplete="off" placeholder="密码"  type="password"  />
					</el-form-item>
					<el-form-item class="list-item" v-if="tableName=='shangjia'">
						<div class="lable" :class="changeRules('mima')?'required':''">确认密码：</div>
						<el-input  v-model="ruleForm.mima2" autocomplete="off" placeholder="确认密码" type="password" />
					</el-form-item>
					<el-form-item class="list-item" v-if="tableName=='shangjia'">
						<div class="lable" :class="changeRules('shangjiaxingming')?'required':''">商家姓名：</div>
						<el-input  v-model="ruleForm.shangjiaxingming"  autocomplete="off" placeholder="商家姓名"  type="text"  />
					</el-form-item>
					<el-form-item class="list-item" v-if="tableName=='shangjia'">
						<div class="lable" :class="changeRules('xingbie')?'required':''">性别：</div>
						<el-select v-model="ruleForm.xingbie" placeholder="请选择性别" >
							<el-option
								v-for="(item,index) in shangjiaxingbieOptions"
								v-bind:key="index"
								:label="item"
								:value="item">
							</el-option>
						</el-select>
					</el-form-item>
					<el-form-item class="list-item" v-if="tableName=='shangjia'">
						<div class="lable" :class="changeRules('shouji')?'required':''">手机：</div>
						<el-input  v-model="ruleForm.shouji"  autocomplete="off" placeholder="手机"  type="text"  />
					</el-form-item>
					<el-form-item class="list-item" v-if="tableName=='shangjia'">
						<div class="lable" :class="changeRules('touxiang')?'required':''">头像：</div>
						<file-upload
							tip="点击上传头像"
							action="file/upload"
							:limit="3"
							:multiple="true"
							:fileUrls="ruleForm.touxiang?ruleForm.touxiang:''"
							@change="shangjiatouxiangUploadChange"
						></file-upload>
					</el-form-item>
					<div class="register-btn">
						<div class="register-btn1">
							<button type="button" class="r-btn" @click="login()">注册</button>
						</div>
						<div class="register-btn2">
							<div class="r-login" @click="close()">已有账号，直接登录</div>
						</div>
					</div>
				</div>
			</el-form>
		</div>
	</div>
</template>

<script>
	import 'animate.css'
export default {
	data() {
		return {
			ruleForm: {
			},
			forgetForm: {},
            pageFlag : '',
			tableName:"",
			rules: {},
            yonghuxingbieOptions: [],
            shangjiaxingbieOptions: [],
		};
	},
	mounted(){
		this.pageFlag = this.$route.query.pageFlag
		if(this.$route.query.pageFlag=='register'){
			
			let table = this.$storage.get("loginTable");
			this.tableName = table;
			if(this.tableName=='yonghu'){
				this.ruleForm = {
					yonghuzhanghao: '',
					mima: '',
					yonghuxingming: '',
					xingbie: '',
					shouji: '',
					touxiang: '',
					money: '',
				}
			}
			if(this.tableName=='shangjia'){
				this.ruleForm = {
					shangjiazhanghao: '',
					mima: '',
					shangjiaxingming: '',
					xingbie: '',
					shouji: '',
					touxiang: '',
					money: '',
				}
			}
			if ('yonghu' == this.tableName) {
				this.rules.yonghuzhanghao = [{ required: true, message: '请输入用户账号', trigger: 'blur' }]
			}
			if ('yonghu' == this.tableName) {
				this.rules.mima = [{ required: true, message: '请输入密码', trigger: 'blur' }]
			}
			if ('yonghu' == this.tableName) {
				this.rules.yonghuxingming = [{ required: true, message: '请输入用户姓名', trigger: 'blur' }]
			}
			if ('shangjia' == this.tableName) {
				this.rules.shangjiazhanghao = [{ required: true, message: '请输入商家账号', trigger: 'blur' }]
			}
			if ('shangjia' == this.tableName) {
				this.rules.mima = [{ required: true, message: '请输入密码', trigger: 'blur' }]
			}
			if ('shangjia' == this.tableName) {
				this.rules.shangjiaxingming = [{ required: true, message: '请输入商家姓名', trigger: 'blur' }]
			}
			this.yonghuxingbieOptions = "男,女".split(',')
			this.shangjiaxingbieOptions = "男,女".split(',')
		}
	},
	created() {
	},
	destroyed() {
		  	},
	methods: {
		changeRules(name){
			if(this.rules[name]){
				return true
			}
			return false
		},
		// 获取uuid
		getUUID () {
			return new Date().getTime();
		},
		close(){
			this.$router.push({ path: "/login" });
		},
        yonghutouxiangUploadChange(fileUrls) {
            this.ruleForm.touxiang = fileUrls;
        },
        shangjiatouxiangUploadChange(fileUrls) {
            this.ruleForm.touxiang = fileUrls;
        },

        // 多级联动参数


		// 注册
		login() {
			var url=this.tableName+"/register";
			if((!this.ruleForm.yonghuzhanghao) && `yonghu` == this.tableName){
				this.$message.error(`用户账号不能为空`);
				return
			}
			if((!this.ruleForm.mima) && `yonghu` == this.tableName){
				this.$message.error(`密码不能为空`);
				return
			}
			if((this.ruleForm.mima!=this.ruleForm.mima2) && `yonghu` == this.tableName){
				this.$message.error(`两次密码输入不一致`);
				return
			}
			if((!this.ruleForm.yonghuxingming) && `yonghu` == this.tableName){
				this.$message.error(`用户姓名不能为空`);
				return
			}
			if(`yonghu` == this.tableName && this.ruleForm.shouji &&(!this.$validate.isMobile(this.ruleForm.shouji))){
				this.$message.error(`手机应输入手机格式`);
				return
			}
            if(this.ruleForm.touxiang!=null) {
                this.ruleForm.touxiang = this.ruleForm.touxiang.replace(new RegExp(this.$base.url,"g"),"");
            }
			if((!this.ruleForm.shangjiazhanghao) && `shangjia` == this.tableName){
				this.$message.error(`商家账号不能为空`);
				return
			}
			if((!this.ruleForm.mima) && `shangjia` == this.tableName){
				this.$message.error(`密码不能为空`);
				return
			}
			if((this.ruleForm.mima!=this.ruleForm.mima2) && `shangjia` == this.tableName){
				this.$message.error(`两次密码输入不一致`);
				return
			}
			if((!this.ruleForm.shangjiaxingming) && `shangjia` == this.tableName){
				this.$message.error(`商家姓名不能为空`);
				return
			}
			if(`shangjia` == this.tableName && this.ruleForm.shouji &&(!this.$validate.isMobile(this.ruleForm.shouji))){
				this.$message.error(`手机应输入手机格式`);
				return
			}
            if(this.ruleForm.touxiang!=null) {
                this.ruleForm.touxiang = this.ruleForm.touxiang.replace(new RegExp(this.$base.url,"g"),"");
            }
			this.$http({
				url: url,
				method: "post",
				data:this.ruleForm
			}).then(({ data }) => {
				if (data && data.code === 0) {
					this.$message({
						message: "注册成功",
						type: "success",
						duration: 1500,
						onClose: () => {
							this.$router.replace({ path: "/login" });
						}
					});
				} else {
					this.$message.error(data.msg);
				}
			});
		}
	}
};
</script>

<style lang="scss" scoped>
.register-container {
	position: relative;
	background: url(http://codegen.caihongy.cn/20240923/485f888d6fd7472b921f33d2bc986e70.jpg) no-repeat center center / 100% 100%;
	background: url(http://codegen.caihongy.cn/20240923/485f888d6fd7472b921f33d2bc986e70.jpg) no-repeat center center / 100% 100%;
	display: flex;
	width: 100%;
	min-height: 100vh;
	justify-content: flex-end;
	align-items: center;
	position: relative !important;
	.rgs-form {
		.rgs-form2 {
		margin: 20px 0;
		width: 100%;
		}
		border-radius: 0;
		padding: 0 6% 0 10%;
		margin: 0;
		z-index: 1000;
		background: #dee9ff;
		display: flex;
		width: 43%;
		min-height: 100vh;
		align-items: center;
		position: relative;
		flex-wrap: wrap;
		height: auto;
		.title {
			padding: 0 80px;
			margin: 0 0 20px 0;
			color: #333;
			background: none;
			font-weight: 600;
			width: 100%;
			font-size: 26px;
			line-height: 40px;
			text-align: left;
		}
		.list-item {
			padding: 0 0 0 0px;
			margin: 0 0 10px;
			width: 100%;
			border-color: #36527d;
			border-width: 0 0 2px;
			position: relative;
			border-style: solid;
			height: auto;
			/deep/ .el-form-item__content {
			}
			.lable {
				padding: 0;
				color: #333;
				left: -130px;
				width: 130px;
				font-size: 16px;
				line-height: 40px;
				position: absolute !important;
				text-align: right;
			}
			.el-input {
				width: 100%;
			}
			.el-input /deep/ .el-input__inner {
				border: 0px solid #e6e6e6;
				border-radius: 0px;
				padding: 0 10px;
				color: #666;
				background: none;
				width: 100%;
				font-size: 16px;
				height: 40px;
			}
			.el-input /deep/ .el-input__inner:focus {
				border: 0px solid #e6e6e6;
				border-radius: 0px;
				padding: 0 10px;
				outline: none;
				color: #666;
				background: none;
				width: 100%;
				font-size: 16px;
				height: 40px;
			}
			.el-input-number {
				width: 100%;
			}
			.el-input-number /deep/ .el-input__inner {
				text-align: center;
				border: 0px solid #e6e6e6;
				border-radius: 0px;
				padding: 0 10px;
				color: #666;
				background: none;
				width: 100%;
				font-size: 16px;
				height: 40px;
			}
			.el-input-number /deep/ .el-input__inner:focus {
				border: 0px solid #e6e6e6;
				border-radius: 0px;
				padding: 0 10px;
				outline: none;
				color: #666;
				background: none;
				width: 100%;
				font-size: 16px;
				height: 40px;
			}
			.el-input-number /deep/ .el-input-number__decrease {
				display: none;
			}
			.el-input-number /deep/ .el-input-number__increase {
				display: none;
			}
			.el-select {
				width: 100%;
			}
			.el-select /deep/ .el-input__inner {
				border: 0px solid #e6e6e6;
				border-radius: 0px;
				padding: 0 10px;
				color: #666;
				background: none;
				width: 100%;
				font-size: 16px;
				height: 38px;
			}
			.el-select /deep/ .el-input__inner:focus {
				border: 0px solid #e6e6e6;
				border-radius: 0px;
				padding: 0 10px;
				outline: none;
				color: #666;
				background: none;
				width: 100%;
				font-size: 16px;
				height: 38px;
			}
			.el-date-editor {
				width: 100%;
			}
			.el-date-editor /deep/ .el-input__inner {
				border: 0px solid #e6e6e6;
				border-radius: 0px;
				padding: 0 10px 0 30px;
				color: #666;
				background: none;
				width: 100%;
				font-size: 16px;
				height: 38px;
			}
			.el-date-editor /deep/ .el-input__inner:focus {
				border: 0px solid #e6e6e6;
				border-radius: 0px;
				padding: 0 10px 0 30px;
				outline: none;
				color: #666;
				background: none;
				width: 100%;
				font-size: 16px;
				height: 38px;
			}
			.el-date-editor.el-input {
				width: 100%;
			}
			/deep/ .el-upload--picture-card {
				background: transparent;
				border: 0;
				border-radius: 0;
				width: auto;
				height: auto;
				line-height: initial;
				vertical-align: middle;
			}
			/deep/ .upload .upload-img {
				border: 2px solid #36527d80;
				cursor: pointer;
				border-radius: 0px;
				color: #36527d;
				background: none;
				width: 90px;
				font-size: 24px;
				line-height: 60px;
				text-align: center;
				height: 60px;
			}
			/deep/ .el-upload-list .el-upload-list__item {
				border: 2px solid #36527d80;
				cursor: pointer;
				border-radius: 0px;
				color: #36527d;
				background: none;
				width: 90px;
				font-size: 24px;
				line-height: 60px;
				text-align: center;
				height: 60px;
			}
			/deep/ .el-upload .el-icon-plus {
				border: 2px solid #36527d80;
				cursor: pointer;
				border-radius: 0px;
				color: #36527d;
				background: none;
				width: 90px;
				font-size: 24px;
				line-height: 60px;
				text-align: center;
				height: 60px;
			}
			/deep/ .el-upload__tip {
				color: #666;
				font-size: 15px;
			}
			/deep/ .el-input__inner::placeholder {
				color: #999;
				font-size: 16px;
			}
			.required {
				position: relative;
			}
			.required::after{
				color: red;
				left: 125px;
				line-height: 42px;
				position: inherit;
				content: "*";
				order: -1;
			}
			.editor {
				background: none;
				width: 100%;
				height: auto;
			}
			.editor>.avatar-uploader {
				line-height: 0;
				height: 0;
			}
		}
		.list-item.email {
			input {
				border: 0px solid #e6e6e6;
				border-radius: 0px 0 0 0px;
				padding: 0 10px;
				margin: 0;
				color: #666;
				background: none;
				flex: 1;
				width: 100%;
				font-size: 14px;
				height: 38px;
			}
			input:focus {
				border: 0px solid #e6e6e6;
				border-radius: 0px 0 0 0px !important;
				padding: 0 10px;
				outline: none;
				color: #666;
				background: none;
				width: 100%;
				font-size: 14px;
				height: 38px;
			}
			input::placeholder {
				color: #999;
				font-size: 16px;
			}
			button {
				border: 0px solid #e6e6e6;
				cursor: pointer;
				border-radius: 0 0px 0px 0;
				padding: 0;
				margin: 0 0 0 10px;
				color: #fff;
				background: #1a3194;
				width: 120px;
				font-size: 15px;
				height: 38px;
			}
			button:hover {
				opacity: 0.8;
			}
		}
		.register-btn {
			display: flex;
			width: 100%;
			flex-wrap: wrap;
		}
		.register-btn1 {
			width: 100%;
			text-align: center;
			order: 3;
		}
		.register-btn2 {
			width: 100%;
		}
		.r-btn {
			border: 15px solid #152981;
			cursor: pointer;
			border-radius: 100%;
			padding: 0 10px;
			margin: 0 auto 10px;
			color: #fff;
			background: #1a3194;
			font-weight: 600;
			width: 130px;
			font-size: 22px;
			min-width: 68px;
			height: 130px;
		}
		.r-btn:hover {
			border: 0px solid rgba(0, 0, 0, 1);
			opacity: 0.8;
		}
		.r-login {
			cursor: pointer;
			padding: 0;
			color: #333;
			display: inline-block;
			width: 100%;
			font-size: 15px;
			line-height: 40px;
			text-align: right;
		}
		.r-login:hover {
			opacity: 1;
		}
	}
}
	
	::-webkit-scrollbar {
	  display: none;
	}
</style>
