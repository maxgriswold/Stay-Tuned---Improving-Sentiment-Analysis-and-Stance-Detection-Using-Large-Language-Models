
dat=read.csv("./data/results/analysis_results.csv",na.strings=c("","NA"))

dat$tune_data=as.character(dat$tune_data)
dat$tune_data[is.na(dat$tune_data)]="XXX"
dat$prompt_name=as.character(dat$prompt_name)
dat$prompt_name[is.na(dat$prompt_name)]="XXX"


dat2=read.csv("./data/processed/pol_tweets_processed.csv")
dat$id=paste(substr(dat$data_name,1,1),formatC(dat$id, width = 5, format = "d", flag = "0"),sep="")
dat2$id=paste("p",formatC(dat2$id, width = 5, format = "d", flag = "0"),sep="")

dat$bioname=dat2[match(dat$id,dat2$id),c("bioname")]
dat$party_code=dat2[match(dat$id,dat2$id),c("party_code")]

dat$train[is.na(dat$train)]=FALSE

dat$method=paste(dat$model_name,dat$tune_data,dat$prompt_name,sep=".")
dat$method=gsub(".XXX","",dat$method,fixed=TRUE)

dats = c("user_val", "pol", "kawintiranon", 'li')

for (dt in dats){
  use.dat=dt
  
  keep=c("id","method","party_code","subject","both_subjects","score","score_nominate","est_score","train")
  dat.tmp=dat[dat$data_name==use.dat,keep]
  meths=levels(factor(dat.tmp$method))
  tuned=grepl("handcode|party|nominate",meths)
  dat1=reshape(dat.tmp,v.names=c("est_score","train"),idvar="id",timevar="method",direction="wide")
  
  here=paste("est_score",meths,sep=".")
  
  here1=paste("train",meths,sep=".")
  
  scr.var="score"
  if(use.dat=='pol'){scr.var="score_nominate"}
  
  dat2=dat1[,c(scr.var,here)]
  dat2=dat2>0
  
  option=1
  if(use.dat=="pol"&option==1){
    dat2[dat1$party_code=="R"&dat1$subject=="trump",]=dat1[dat1$party_code=="R"&dat1$subject=="trump",c(scr.var,here)]>0
    dat2[dat1$party_code=="D"&dat1$subject=="trump",]=dat1[dat1$party_code=="D"&dat1$subject=="trump",c(scr.var,here)]>=0
    dat2[dat1$party_code=="R"&dat1$subject=="biden",]=dat1[dat1$party_code=="R"&dat1$subject=="biden",c(scr.var,here)]>=0
    dat2[dat1$party_code=="D"&dat1$subject=="biden",]=dat1[dat1$party_code=="D"&dat1$subject=="biden",c(scr.var,here)]>0
  } else if(use.dat=="pol"&option == 2) {
    dat2[dat1$party_code=="R"&dat1$subject=="trump",]=dat1[dat1$party_code=="R"&dat1$subject=="trump",c(scr.var,here)]>=0
    dat2[dat1$party_code=="D"&dat1$subject=="trump",]=dat1[dat1$party_code=="D"&dat1$subject=="trump",c(scr.var,here)]>0
    dat2[dat1$party_code=="R"&dat1$subject=="biden",]=dat1[dat1$party_code=="R"&dat1$subject=="biden",c(scr.var,here)]>0
    dat2[dat1$party_code=="D"&dat1$subject=="biden",]=dat1[dat1$party_code=="D"&dat1$subject=="biden",c(scr.var,here)]>=0
  } else if(use.dat=="pol"&option == 3) {
    dat2[dat1$party_code=="R"&dat1$subject=="trump",]=dat1[dat1$party_code=="R"&dat1$subject=="trump",c(scr.var,here)]>=0
    dat2[dat1$party_code=="D"&dat1$subject=="trump",]=dat1[dat1$party_code=="D"&dat1$subject=="trump",c(scr.var,here)]>=0
    dat2[dat1$party_code=="R"&dat1$subject=="biden",]=dat1[dat1$party_code=="R"&dat1$subject=="biden",c(scr.var,here)]>=0
    dat2[dat1$party_code=="D"&dat1$subject=="biden",]=dat1[dat1$party_code=="D"&dat1$subject=="biden",c(scr.var,here)]>=0
  } else if(use.dat=="pol"&option == 4) {
    dat2[dat1$party_code=="R"&dat1$subject=="trump",]=dat1[dat1$party_code=="R"&dat1$subject=="trump",c(scr.var,here)]>0
    dat2[dat1$party_code=="D"&dat1$subject=="trump",]=dat1[dat1$party_code=="D"&dat1$subject=="trump",c(scr.var,here)]>0
    dat2[dat1$party_code=="R"&dat1$subject=="biden",]=dat1[dat1$party_code=="R"&dat1$subject=="biden",c(scr.var,here)]>0
    dat2[dat1$party_code=="D"&dat1$subject=="biden",]=dat1[dat1$party_code=="D"&dat1$subject=="biden",c(scr.var,here)]>0
  } 
  
  dat3=dat1[,c(scr.var,here)]
  for(i in 1:NCOL(dat3)) {
    dat3[,i]=cut(dat3[,i],c(-Inf,-.3,.3,Inf),labels=c("Negative","Neutral","Positive"))
  }
  levs=levels(dat3[,scr.var])
  
  subs=levels(factor(dat1$subject))
  
  if(use.dat=="user_val") {
    use=list(both=dat1$both_subjects,one=!dat1$both_subjects,all=rep(TRUE,NROW(dat1)))
  } else {
    use=list(all=rep(TRUE,NROW(dat1)))
  }
  
  use.t=as.matrix(dat1[,here1])
  
  corrs=rmses=maes=array(NA,c(length(here),length(subs),length(use)))
  dimnames(corrs)=dimnames(rmses)=dimnames(maes)=list(meths,subs,names(use))
  dimnames(rmses)[[2]]=paste("rMSE",subs,sep=".")
  dimnames(maes)[[2]]=paste("MAE",subs,sep=".")
  corrs=array(NA,c(length(here),5*length(subs),length(use)))
  dimnames(corrs)=list(meths,paste(c(rep(subs[1],5),rep(subs[2],5)),rep(c("rMSE","MAE","lower","r","upper"),2),sep="."),names(use))
  corrs.b=array(NA,c(length(here),9*length(subs),length(use)))
  dimnames(corrs.b)=list(meths,paste(c(rep(subs[1],9),rep(subs[2],9)),rep(c("POS","NEG","lower","r","upper","Accuracy","Precision","Recall","F1"),2),sep="."),names(use))
  corrs.c=array(NA,c(length(here),(length(levs)+2)*length(subs),length(use)))
  dimnames(corrs.c)=list(meths,paste(c(rep(subs[1],length(levs)+2),rep(subs[2],length(levs)+2)),rep(c(paste("Cor.",levs,sep=""),"ChiSq","CramerV"),2),sep="."),names(use))
  
  for(i in 1:length(subs)){
    for(j in 1:length(use)){
      dat1.tmp=as.matrix(dat1[,here])
      dat1.tmp[use.t]=NA
      use.tmp=use[[j]]&dat1$subject==subs[i]
      maes[,i,j]=colMeans(abs(dat1.tmp[use.tmp,]-dat1[use.tmp,scr.var]),na.rm=TRUE)
      rmses[,i,j]=sqrt(colMeans((dat1.tmp[use.tmp,]-dat1[use.tmp,scr.var])^2,na.rm=TRUE))
      
      corrs[,2+(i-1)*5,j]=colMeans(abs(dat1.tmp[use.tmp,]-dat1[use.tmp,scr.var]),na.rm=TRUE)
      corrs[,1+(i-1)*5,j]=sqrt(colMeans((dat1.tmp[use.tmp,]-dat1[use.tmp,scr.var])^2,na.rm=TRUE))
      corr.tmp=psych::corr.test(dat1.tmp[use.tmp,],dat1[use.tmp,scr.var])
      corrs[,(3:5)+(i-1)*5,j]=as.matrix(corr.tmp$ci)[,-4]
      
      use.tmp1=use.tmp&dat2[,scr.var]
      use.tmp2=use.tmp&!dat2[,scr.var]
      
      posneg=cbind(
        POS=colMeans(dat2[use.tmp1,here]),
        NEG=colMeans(dat2[use.tmp2,here])
      )
      corrs.b[,(1:2)+(i-1)*9,j]=posneg
      
      corr.tmp=psych::corr.test(dat2[use.tmp,here],dat2[use.tmp,scr.var])
      corrs.b[,(3:5)+(i-1)*9,j]=as.matrix(corr.tmp$ci)[,-4]
      
      tp=colMeans(dat2[use.tmp,here]&dat2[use.tmp,scr.var],na.rm=TRUE)
      tn=colMeans(!(dat2[use.tmp,here])&!(dat2[use.tmp,scr.var]),na.rm=TRUE)
      fp=colMeans(!(dat2[use.tmp,here])&dat2[use.tmp,scr.var],na.rm=TRUE)
      fn=colMeans(dat2[use.tmp,here]&!(dat2[use.tmp,scr.var]),na.rm=TRUE)
      recall=tp/(tp+fp)
      precision=tp/(tp+fn)
      accuracy=(tp+tn)/(tp+tn+fp+fn)
      fscore=2*tp/(2*tp+fp+fn)
      corrs.b[,6+(i-1)*9,j]=accuracy
      corrs.b[,7+(i-1)*9,j]=precision
      corrs.b[,8+(i-1)*9,j]=recall
      corrs.b[,9+(i-1)*9,j]=fscore
      
      
      x=dat3[use.tmp,scr.var]
      y=dat3[use.tmp,here]
      
      cors=NULL
      chisq=0
      
      for(j1 in 1:length(levs)) {
        for(i1 in 1:length(levs)) {
          x.tmp=x==levs[i1]
          y.tmp=y==levs[j1]
          
          chisq=chisq+colSums(!is.na(y))*(colMeans(x.tmp&y.tmp,na.rm=TRUE)-mean(x.tmp,na.rm=TRUE)*colMeans(y.tmp,na.rm=TRUE))^2/(mean(x.tmp,na.rm=TRUE)*colMeans(y.tmp,na.rm=TRUE))
          
          if(j1==i1){cors=cbind(cors,cor(y.tmp,x.tmp,use="pairwise.complete.obs"))}
        }
      }
      
      colnames(cors)=paste("Cor.",levs,sep="")
      
      corrs.c[,1:NCOL(cors)+(i-1)*(NCOL(cors)+2),j]=cors
      
      corrs.c[,(NCOL(cors)+1)+(i-1)*(NCOL(cors)+2),j]=chisq
      corrs.c[,(NCOL(cors)+2)+(i-1)*(NCOL(cors)+2),j]=sqrt(chisq/sum(!is.na(x))/(length(levs)-1))
      
      
    }}
  
  corrs.c[is.nan(corrs.c)]=NA
  
  if(use.dat=="user_val") {
    out=list(
      corrs[,,1],
      corrs[,,2],
      corrs[,,3],
      corrs.b[,,1],
      corrs.b[,,2],
      corrs.b[,,3],
      corrs.c[,,1],
      corrs.c[,,2],
      corrs.c[,,3]
    )
    names(out)=c(paste("Continuous, ",dimnames(corrs)[[3]],sep=""),paste("Binary, ",dimnames(corrs.b)[[3]],sep=""),paste("Categorical, ",dimnames(corrs.c)[[3]],sep=""))
  } else {
    out=list(
      corrs[,,1],
      corrs.b[,,1],
      corrs.c[,,1]
    )
    names(out)=c("Continuous","Binary","Categorical")
  }
  
  out.results1=function(results,file="NonResponseRates.xlsx",use.xlsx=TRUE,na="",array.dim=3)
  {
    if(class(results)=="array"){is.array=TRUE;a.nams=dimnames(results)[[array.dim]]}
    else{is.array=FALSE;a.nams=names(results)}
    
    file=gsub(".xlsx","",file)
    file=gsub(".csv","",file)
    file=gsub(".xls","",file)
    if(class(results)!="list"&class(results)!="array"){results=list(results);names(results)=""}
    
    if(use.xlsx){
      #require(xlsx)
      require(openxlsx)
      file=paste(file,".xlsx",sep="")
      wb=createWorkbook()
      
      for(i in 1:length(a.nams))
      {
        if(!is.array){out <- results[[i]]}
        else if(array.dim==1){out <- results[i,,]}
        else if(array.dim==2){out <- results[,i,]}
        else{out <- results[,,i]}
        nams <- colnames(out)
        r.nams <- rownames(out)
        p <- NCOL(out)
        n <- NROW(out)
        
        nam <- a.nams[i]
        sheet <- addWorksheet(wb, sheetName=nam)
        setColWidths(wb, i, cols=1:1, widths = 13)
        
        writeData(wb, sheet = i, out, rowNames = TRUE, colNames=TRUE)
        sty <- createStyle(numFmt = "0.000%")
        addStyle(wb, i, style = sty, rows = 2:11, cols = which(grepl("freq",colnames(out)))+1, gridExpand = TRUE)
        
      }
      
      saveWorkbook(wb,file=file,overwrite=TRUE)
      rm(wb)
    }
    else
    {
      for(i in 1:length(results))
      {
        file.tmp=paste(file,"_",a.nams[i],".csv",sep="")
        if(!is.array){write.csv(results[[i]],file.tmp,na=na)}
        else if(array.dim==1){write.csv(results[i,,],file.tmp,na=na)}
        else if(array.dim==2){write.csv(results[,i,],file.tmp,na=na)}
        else{write.csv(results[,,i],file.tmp,na=na)}
      }
    }
    
  }
  
  out.results1(out, sprintf("./data/results/summary_statistics/%s_summary_statistics.xlsx", use.dat))
}

