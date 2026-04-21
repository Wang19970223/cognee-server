# Spring 事务失效排查指南

## 场景描述
在 Spring Boot 项目中，使用 `@Transactional` 注解的方法事务不生效，数据库操作未回滚。

## 常见原因

### 1. 同类内部调用
同一个类中，方法 A 调用带 `@Transactional` 的方法 B，事务不生效。
Spring 事务基于 AOP 代理，内部调用绕过了代理对象。

**解决方案**：通过注入自身 Bean 来调用，或使用 `AopContext.currentProxy()`。

### 2. 方法非 public
`@Transactional` 只对 `public` 方法生效。

### 3. 异常被吞掉
方法内 `try-catch` 捕获了异常但未重新抛出，事务不会回滚。

**解决方案**：catch 块中加 `TransactionAspectSupport.currentTransactionStatus().setRollbackOnly()`。

### 4. 数据库引擎不支持事务
MySQL 使用 MyISAM 引擎不支持事务，需切换为 InnoDB。

## 排查步骤
1. 检查方法是否为 public
2. 检查是否存在内部调用
3. 确认异常是否被正确传播
4. 查看数据库引擎类型
5. 开启 Spring 事务日志：`logging.level.org.springframework.transaction=DEBUG`
