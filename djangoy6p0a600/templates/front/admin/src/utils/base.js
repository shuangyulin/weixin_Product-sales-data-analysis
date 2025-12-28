const base = {
    get() {
        return {
            url : "http://localhost:8080/djangoy6p0a600/",
            name: "djangoy6p0a600",
            // 退出到首页链接
            indexUrl: 'http://localhost:8080/front/h5/index.html'
        };
    },
    getProjectName(){
        return {
            projectName: "基于微信小程序的直播带货商品数据分析系统的设计与实现"
        } 
    }
}
export default base
