dat=fread("C:/Users/griswold/Documents/GitHub/twitter-representative-pop/public_facing/data/results/stance_analysis_results_pol.csv")
dat$id1=paste0(dat$id,dat$bioname)
dat$id2=paste0(dat$text,dat$bioname)

dat$method=gsub(" ",".",dat$method,fixed=TRUE)

dat1=reshape(dat,v.names=c("sentiment_tweet","sentiment_person"),idvar="id1",timevar="method",direction="wide")

is.rt=substr(dat1$text,1,2)=="RT"

here=colnames(dat1)[grepl("sentiment_tweet",colnames(dat1))]
here1=gsub("sentiment_tweet.","",here,fixed=TRUE)

dat1$party_code=relevel(factor(dat1$party_code),ref="R")

dat.t=data.frame(dat1[dat1$subject_name=="Trump",c("bioname","party_code",here)])
dat.b=data.frame(dat1[dat1$subject_name=="Biden",c("bioname","party_code",here)])

dat1.t=dat.t
dat1.b=dat.b

### create binary outcome
option = 2

if(option==1){
dat1.t[dat1.t$party_code=="R",3:NCOL(dat.t)]=dat1.t[dat1.t$party_code=="R",3:NCOL(dat.t)]>0
dat1.t[dat1.t$party_code=="D",3:NCOL(dat.t)]=dat1.t[dat1.t$party_code=="D",3:NCOL(dat.t)]>=0
dat1.b[dat1.b$party_code=="R",3:NCOL(dat.b)]=dat1.b[dat1.b$party_code=="R",3:NCOL(dat.b)]>=0
dat1.b[dat1.b$party_code=="D",3:NCOL(dat.b)]=dat1.b[dat1.b$party_code=="D",3:NCOL(dat.b)]>0
} else if(option == 2) {
dat1.t[dat1.t$party_code=="R",3:NCOL(dat.t)]=dat1.t[dat1.t$party_code=="R",3:NCOL(dat.t)]>=0
dat1.t[dat1.t$party_code=="D",3:NCOL(dat.t)]=dat1.t[dat1.t$party_code=="D",3:NCOL(dat.t)]>0
dat1.b[dat1.b$party_code=="R",3:NCOL(dat.b)]=dat1.b[dat1.b$party_code=="R",3:NCOL(dat.b)]>0
dat1.b[dat1.b$party_code=="D",3:NCOL(dat.b)]=dat1.b[dat1.b$party_code=="D",3:NCOL(dat.b)]>=0
} else if(option == 3) {
dat1.t[dat1.t$party_code=="R",3:NCOL(dat.t)]=dat1.t[dat1.t$party_code=="R",3:NCOL(dat.t)]>=0
dat1.t[dat1.t$party_code=="D",3:NCOL(dat.t)]=dat1.t[dat1.t$party_code=="D",3:NCOL(dat.t)]>=0
dat1.b[dat1.b$party_code=="R",3:NCOL(dat.b)]=dat1.b[dat1.b$party_code=="R",3:NCOL(dat.b)]>=0
dat1.b[dat1.b$party_code=="D",3:NCOL(dat.b)]=dat1.b[dat1.b$party_code=="D",3:NCOL(dat.b)]>=0
} else if(option == 4) {
dat1.t[dat1.t$party_code=="R",3:NCOL(dat.t)]=dat1.t[dat1.t$party_code=="R",3:NCOL(dat.t)]>0
dat1.t[dat1.t$party_code=="D",3:NCOL(dat.t)]=dat1.t[dat1.t$party_code=="D",3:NCOL(dat.t)]>0
dat1.b[dat1.b$party_code=="R",3:NCOL(dat.b)]=dat1.b[dat1.b$party_code=="R",3:NCOL(dat.b)]>0
dat1.b[dat1.b$party_code=="D",3:NCOL(dat.b)]=dat1.b[dat1.b$party_code=="D",3:NCOL(dat.b)]>0
}

here2=colnames(dat.t)[-c(1,2)]

### Output for: Linear model with continuous outcome
cont.lm=NULL

### Output for: Linear model with random effects and continuous outcome
cont.re=NULL

### Output for: Welch stats
cont.w=NULL
bin.w=NULL

### Output for: Correlations
corr.cont=NULL
corr.bin=NULL

### Output for: Fisher's exact test
fisher=NULL

### Output for: Chisq test
chisq=NULL

### Output for: Logistic model with binary outcome
bin.glm=NULL

### Output for: Logistic model with random effects and binary outcome
bin.re=NULL

for(i in 1:length(here)) {

form=paste(here2[i],"~","party_code")
form=as.formula(form)

form1=paste(here2[i],"~","party_code","+ (1|bioname)")
form1=as.formula(form1)

tmp1.t=dat.t[dat.t$party_code=="D",here2[i]]
tmp2.t=dat.t[dat.t$party_code=="R",here2[i]]
tmp1.b=dat.b[dat.b$party_code=="D",here2[i]]
tmp2.b=dat.b[dat.b$party_code=="R",here2[i]]

test.t=t.test(tmp1.t,tmp2.t)
test.b=t.test(tmp1.b,tmp2.b)
tmp.w=c(test.t[[5]],Est.T=test.t[[5]][1]-test.t[[5]][2],test.t$conf.int,Stat.T=test.t$statistic,
test.b[[5]],Est.B=test.b[[5]][1]-test.b[[5]][2],test.b$conf.int,Stat.B=test.b$statistic
)
names(tmp.w)=rep(c("Mean Dem","Mean Republican","Mean Difference","CI 0.025","CI 0.975","T-stat"),2)

lmmod.t=lm(form,data=dat.t)
lmmod.b=lm(form,data=dat.b)
coef.t=c(summary(lmmod.t)$coefficients[2,c(1,3)],confint(lmmod.t)[2,])
coef.b=c(summary(lmmod.b)$coefficients[2,c(1,3)],confint(lmmod.b)[2,])
#names(coef.t)=c("Est.T","Stat.T","pval.T")
#names(coef.b)=c("Est.B","Stat.B","pval.B")

cont.lm=rbind(cont.lm,c(coef.t[c(1,3,4,2)],coef.b[c(1,3,4,2)]))
cont.w=rbind(cont.w,tmp.w)

remod.t=lme4::lmer(form1,data=dat.t)
remod.b=lme4::lmer(form1,data=dat.b)
mod.t=c(summary(remod.t)$coefficients[2,c(1,3)],confint(remod.t)[4,])
mod.b=c(summary(remod.b)$coefficients[2,c(1,3)],confint(remod.b)[4,])

cont.re=rbind(cont.re,c(mod.t[c(1,3,4,2)],mod.b[c(1,3,4,2)]))

tab.t=table(dat1.t$party_code=="R",dat1.t[,here2[i]])
tab.b=table(dat1.b$party_code=="D",dat1.b[,here2[i]])

tmp1.t=dat1.t[dat1.t$party_code=="D",here2[i]]
tmp2.t=dat1.t[dat1.t$party_code=="R",here2[i]]
tmp1.b=dat1.b[dat1.b$party_code=="D",here2[i]]
tmp2.b=dat1.b[dat1.b$party_code=="R",here2[i]]

test.t=t.test(tmp1.t,tmp2.t)
test.b=t.test(tmp1.b,tmp2.b)
tmp.bin=c(test.t[[5]],Est.T=test.t[[5]][1]-test.t[[5]][2],test.t$conf.int,Stat.T=test.t$statistic,
test.b[[5]],Est.B=test.b[[5]][1]-test.b[[5]][2],test.b$conf.int,Stat.B=test.b$statistic
)
names(tmp.bin)=rep(c("Mean Dem","Mean Republican","Mean Difference","CI 0.025","CI 0.975","T-stat"),2)
bin.w=rbind(bin.w,tmp.bin)

tmp2=chisq.test(tab.t)
tmp3=chisq.test(tab.b)
chisq.tmp=c(Stat.T=tmp2$statistic,pval.T=tmp2$p.value,Stat.B=tmp3$statistic,pval.B=tmp3$p.value)
chisq=rbind(chisq,chisq.tmp)

tmp2=fisher.test(tab.t)
tmp3=fisher.test(tab.b)
fisher.tmp=c(Stat.T=tmp2$estimate,pval.T=tmp2$p.value,Stat.B=tmp3$estimate,pval.B=tmp3$p.value)
fisher=rbind(fisher,fisher.tmp)

corr.t=psych::corr.test(dat.t[,here2[i]],dat.t$party_code=="R")
corr.b=psych::corr.test(dat.b[,here2[i]],dat.b$party_code=="R")

#corr.tmp=c(Corr.T=psych::phi(tab.t,digits=10),Corr.B=psych::phi(tab.b,digits=10))
corr.tmp=c(as.matrix(corr.t$ci)[,-4],as.matrix(corr.b$ci)[,-4])[c(2,1,3,5,4,6)]
corr.cont=rbind(corr.cont,corr.tmp)

corr.t=psych::corr.test(dat1.t[,here2[i]],dat1.t$party_code=="R")
corr.b=psych::corr.test(dat1.b[,here2[i]],dat1.b$party_code=="R")

#corr.tmp=c(Corr.T=psych::phi(tab.t,digits=10),Corr.B=psych::phi(tab.b,digits=10))
corr.tmp=c(as.matrix(corr.t$ci)[,-4],as.matrix(corr.b$ci)[,-4])[c(2,1,3,5,4,6)]
corr.bin=rbind(corr.bin,corr.tmp)

coef.t=summary(glm(form,data=dat1.t),family=binomial())$coefficients[2,c(1,3,4)]
coef.b=summary(glm(form,data=dat1.b),family=binomial())$coefficients[2,c(1,3,4)]
names(coef.t)=c("Est.T","Stat.T","pval.T")
names(coef.b)=c("Est.B","Stat.B","pval.B")

bin.glm=rbind(bin.glm,c(coef.t,coef.b))

remod.t=lme4::glmer(form1,data=dat1.t,family=binomial)
remod.b=lme4::glmer(form1,data=dat1.b,family=binomial)
mod.t=c(summary(remod.t)$coefficients[2,c(1,3)],confint(remod.t)[3,])
mod.b=c(summary(remod.b)$coefficients[2,c(1,3)],confint(remod.b)[3,])

bin.re=rbind(cont.re,c(mod.t[c(1,3,4,2)],mod.b[c(1,3,4,2)]))

mod.t=summary(lme4::glmer(form1,data=dat1.t,family=binomial))$coefficients[2,c(1,3)]
mod.b=summary(lme4::glmer(form1,data=dat1.b,family=binomial))$coefficients[2,c(1,3)]

bin.re=rbind(bin.re,c(mod.t,mod.b))

}

here3=gsub("."," ",here1,fixed=TRUE)
rownames(corr.cont)=rownames(corr.bin)=rownames(fisher)=rownames(chisq)=rownames(bin.re)=rownames(bin.glm)=rownames(cont.re)=rownames(cont.lm)=rownames(bin.w)=rownames(cont.w)=here1
